#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using namespace std;
using nlohmann::json;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
	vector<double> map_waypoints_x;
	vector<double> map_waypoints_y;
	vector<double> map_waypoints_s;
	vector<double> map_waypoints_dx;
	vector<double> map_waypoints_dy;

	// Waypoint map to read from
	string map_file_ = "../data/highway_map.csv";
	// The max s value before wrapping around the track back to 0
	double max_s = 6945.554;

	std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

	string line;
	while (getline(in_map_, line)) {
		std::istringstream iss(line);
		double x;
		double y;
		float s;
		float d_x;
		float d_y;
		iss >> x;
		iss >> y;
		iss >> s;
		iss >> d_x;
		iss >> d_y;
		map_waypoints_x.push_back(x);
		map_waypoints_y.push_back(y);
		map_waypoints_s.push_back(s);
		map_waypoints_dx.push_back(d_x);
		map_waypoints_dy.push_back(d_y);
	}
	// Start in the middle lane(1)
	int lane = 1;

	// Have a reference velocity to target
	double ref_vel = 0.0; //mph

	h.onMessage([&ref_vel, &lane, &map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
		&map_waypoints_dx, &map_waypoints_dy]
		(uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
			uWS::OpCode opCode) {
		// "42" at the start of the message means there's a websocket message event.
		// The 4 signifies a websocket message
		// The 2 signifies a websocket event
		if (length && length > 2 && data[0] == '4' && data[1] == '2') {

			auto s = hasData(data);

			if (s != "") {
				auto j = json::parse(s);

				string event = j[0].get<string>();

				if (event == "telemetry") {
					// j[1] is the data JSON object

					// Main car's localization Data
					double car_x = j[1]["x"];
					double car_y = j[1]["y"];
					double car_s = j[1]["s"];
					double car_d = j[1]["d"];
					double car_yaw = j[1]["yaw"];
					double car_speed = j[1]["speed"];

					// Previous path data given to the Planner
					auto previous_path_x = j[1]["previous_path_x"];
					auto previous_path_y = j[1]["previous_path_y"];
					// Previous path's end s and d values 
					double end_path_s = j[1]["end_path_s"];
					double end_path_d = j[1]["end_path_d"];

					// Sensor Fusion Data, a list of all other cars on the same side 
					//   of the road.
					auto sensor_fusion = j[1]["sensor_fusion"];

					json msgJson;

					/**
					 *   Define a path made up of (x,y) points that the car will visit
					 *   sequentially every .02 seconds
					 */

					int prev_size = previous_path_x.size();

					/**
					 *   Avoid collision
					 */

					 // Set the safty distance in both lateral and longitudinal directions
					int safe_dist_lat = 2;//m
					int safe_dist_long = 30;//m

					if (prev_size > 0) {
						car_s = end_path_s;
					}

					bool too_close = false;

					// Find ref_v to use
					for (int i = 0; i < sensor_fusion.size(); i++) {
						// Car is in my lane
						float d = sensor_fusion[i][6];
						if (d < (2 + 4 * lane + safe_dist_lat) && d >(2 + 4 * lane - safe_dist_lat)) {
							double vx = sensor_fusion[i][3];
							double vy = sensor_fusion[i][4];
							double check_speed = sqrt(vx * vx + vy * vy);
							double check_car_s = sensor_fusion[i][5];

							check_car_s += ((double)prev_size*.02*check_speed);// project s value outwards in time

							// Check s values greater than mine and safty distance
							if ((check_car_s > car_s) && ((check_car_s - car_s) < safe_dist_long)) {
								too_close = true;
							}
						}
					}
							if (too_close) {
								ref_vel -= .224;// Decelerating
							}
							else if (ref_vel < 49.5) {
								ref_vel += .224;// Accelerating
							}

							// Change lane
							vector<int> reachable_lane;

							if (too_close) {
								// Get lanes which are reachable with just one lane change,
								// including my current lane
								for (int l = 0; l < 3; l++) {
									if (abs(l - lane) < 2) {
										reachable_lane.push_back(l);
									}
								}

								// Variables for the velocities driven on each lane
								vector<int> lane_speeds_0;
								vector<int> lane_speeds_1;
								vector<int> lane_speeds_2;

								// Clearnce flag for a lane change to this lane
								bool lane_clearance_0 = true;
								bool lane_clearance_1 = true;
								bool lane_clearance_2 = true;

								// Getting information about the ego vehicles environment
								for (int i = 0; i < sensor_fusion.size(); i++) {
									// iterate through all vehicles and calculate mag. velocity
									float d = sensor_fusion[i][6];
									for (int l = 0; l < reachable_lane.size(); l++) {

										// I'm only interested of the other lines which I can reach
										// with just one lane change
										if (d < (2 + 4 * reachable_lane[l] + safe_dist_lat) && d >(2 + 4 * reachable_lane[l] - safe_dist_lat)) {

											// get vehicle speed x and y values of the target vehicle
											double vx = sensor_fusion[i][3];
											double vy = sensor_fusion[i][4];

											// get the magnitude speed of the target vehicle
											double check_speed = sqrt(vx * vx + vy * vy);

											// get the s position of the target vehicle
											double check_car_s = sensor_fusion[i][5];
											check_car_s += ((double)prev_size * .02 * check_speed);

											/*  At this point I have the data of one car, which is
												on one of the possible lanes. I know it's velocity,
												but I don't know if it is in front of me or
												behind me. But I know that I'm in this function,
												because my current target vehicle in my lane is
												slower than me.
											*/

											// Generate some information about my environment:
											// looking on the reachable lanes for vehicles within
											// an area from ega vehcile till 50 in front
											if ((check_car_s > car_s) && (check_car_s - car_s < 50)) {
												if (reachable_lane[l] == 0) {
													lane_speeds_0.push_back(check_speed);
												}
												else if (reachable_lane[l] == 1) {
													lane_speeds_1.push_back(check_speed);
												}
												else if (reachable_lane[l] == 2) {
													lane_speeds_2.push_back(check_speed);
												}
											}

											// Is a lane change to the desired lane possible?
											if ((check_car_s > (car_s - 8)) && (check_car_s - car_s < safe_dist_long)) {
												if (reachable_lane[l] == 0) {
													lane_clearance_0 = false;
												}
												else if (reachable_lane[l] == 1) {
													lane_clearance_1 = false;
												}
												else if (reachable_lane[l] == 2) {
													lane_clearance_2 = false;
												}
											}
										}
									}
								}

								/*  Getting now the fastest lane:
									1. Is there a clearnce to change to that lane?
									2. Is the lane free?
									3. If the lane is not free: take that the lane where the
									   3rd slowest car is driving.
								*/
								double best_speed;

								// Iterate trough all lanes and take the best option.
								for (int l = 0; l < reachable_lane.size(); l++) {
									if ((reachable_lane[l] == 0) && lane_clearance_0) {
										if (lane_speeds_0.size() == 0) {
											lane = 0;
										}
										else {
											best_speed = *min_element(lane_speeds_0.begin(), lane_speeds_0.end());
											lane = 0;
										}
									}
									else if ((reachable_lane[l] == 1) && lane_clearance_1) {
										if (lane_speeds_1.size() == 0) {
											lane = 1;
										}
										else {
											double min_speed = *min_element(lane_speeds_1.begin(), lane_speeds_1.end());
											if (best_speed < min_speed) {
												best_speed = min_speed;
												lane = 1;
											}
										}
									}
									else if ((reachable_lane[l] == 2) && lane_clearance_2) {
										if (lane_speeds_2.size() == 0) {
											lane = 2;
										}
										else {
											double min_speed = *min_element(lane_speeds_2.begin(), lane_speeds_2.end());
											if (best_speed < min_speed) {
												best_speed = min_speed;
												lane = 2;
											}
										}
									}
								}
							}
							// Create a list of widely spaced (x,y) waypoints, evenly spaced at 30m
							// Later we will interpolate these waypoints with a spline and fill it in with more points that control spline
							vector<double> ptsx;
							vector<double> ptsy;

							double ref_x = car_x;
							double ref_y = car_y;
							double ref_yaw = deg2rad(car_yaw);

							// If previous size is almost empty, use the car as starting reference
							if (prev_size < 2) {
								// Use two points that make the path tangent to the car
								double prev_car_x = car_x - cos(car_yaw);
								double prev_car_y = car_y - sin(car_yaw);

								ptsx.push_back(prev_car_x);
								ptsx.push_back(car_x);

								ptsy.push_back(prev_car_y);
								ptsy.push_back(car_y);
							}
							// Use the previous path's end point as starting reference
							else {
								// Redefine reference state as previous path end point
								ref_x = previous_path_x[prev_size - 1];
								ref_y = previous_path_y[prev_size - 1];

								double ref_x_prev = previous_path_x[prev_size - 2];
								double ref_y_prev = previous_path_y[prev_size - 2];
								ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

								// Use two points that make the path tangent to the previous path's end point
								ptsx.push_back(ref_x_prev);
								ptsx.push_back(ref_x);

								ptsy.push_back(ref_y_prev);
								ptsy.push_back(ref_y);
							}

							// In Frenet add evenly 30m spaced points ahead of the starting point
							vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
							vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
							vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

							ptsx.push_back(next_wp0[0]);
							ptsx.push_back(next_wp1[0]);
							ptsx.push_back(next_wp2[0]);

							ptsy.push_back(next_wp0[1]);
							ptsy.push_back(next_wp1[1]);
							ptsy.push_back(next_wp2[1]);

							for (int i = 0; i < ptsx.size(); i++) {
								// Shift car reference angle to 0 degrees
								double shift_x = ptsx[i] - ref_x;
								double shift_y = ptsy[i] - ref_y;

								ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
								ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
							}

							// Creat a spline
							tk::spline s;

							// Set (x,y) points to the spline
							s.set_points(ptsx, ptsy);

							// Define the actual (x,y) points we will use for the planner
							vector<double> next_x_vals;
							vector<double> next_y_vals;

							// Start with all of the previous path points from last time
							for (int i = 0; i < previous_path_x.size(); i++) {
								next_x_vals.push_back(previous_path_x[i]);
								next_y_vals.push_back(previous_path_y[i]);
							}

							// Calculate how to break up spline points so we travel at our desired reference velocity
							double target_x = 30.0;
							double target_y = s(target_x);
							double target_dist = sqrt((target_x) * (target_x)+(target_y) * (target_y));

							double x_add_on = 0;

							// Fill up the rest of our path planner after filling it with previous points
							for (int i = 1; i <= 50 - previous_path_x.size(); i++) {

								double N = (target_dist / (.02*ref_vel / 2.24));
								double x_point = x_add_on + target_x / N;
								double y_point = s(x_point);

								x_add_on = x_point;

								double x_ref = x_point;
								double y_ref = y_point;

								// Rotate back after rotaining it earlier
								x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
								y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

								x_point += ref_x;
								y_point += ref_y;

								next_x_vals.push_back(x_point);
								next_y_vals.push_back(y_point);
							}
						
					

							msgJson["next_x"] = next_x_vals;
							msgJson["next_y"] = next_y_vals;

							auto msg = "42[\"control\"," + msgJson.dump() + "]";

							ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
						}  // end "telemetry" if
					}
					else {
						// Manual driving
						std::string msg = "42[\"manual\",{}]";
						ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
					}
				}  // end websocket if
			}); // end h.onMessage

	h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
		std::cout << "Connected!!!" << std::endl;
	});

	h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
		char *message, size_t length) {
		ws.close();
		std::cout << "Disconnected" << std::endl;
	});

	int port = 4567;
	if (h.listen(port)) {
		std::cout << "Listening to port " << port << std::endl;
	}
	else {
		std::cerr << "Failed to listen to port" << std::endl;
		return -1;
	}

	h.run();
   }