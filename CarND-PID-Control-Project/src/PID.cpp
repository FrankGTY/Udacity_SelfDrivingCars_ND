#include "PID.h"

using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
  
  prev_cte = 0.0;
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
}

void PID::UpdateError(double cte) {
  p_error = cte;
  d_error = cte - prev_cte; 
  i_error += cte;
  prev_cte = cte;
}

double PID::TotalError() {
  return -Kp * p_error - Ki * i_error - Kd * d_error;
}