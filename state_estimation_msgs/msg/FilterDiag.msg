# One shadow-EKF update (plan section 9.4). Published by shadow_ekf1 on /shadow_ekf1/diagnostics.
std_msgs/Header header        # stamp = the measurement stamp
string sensor                 # 'speed' or 'gyro'
float64 measurement
float64 innovation            # y - C mu (before the update)
float64 innovation_variance   # S = C Sigma C^T + R
float64 nis                   # innovation^2 / S  (about 1 on average for a consistent filter)
float64 dt                    # predict step that preceded this update [s]
float64[5] sigma_diag         # diag(Sigma) after the update: x, y, theta, v, omega
uint32 dropped_old            # messages older than the filter time (cumulative)
uint32 substeps               # predicts split into sub-steps (cumulative)
uint32 resets                 # filter resets after long gaps (cumulative)
uint32 stale_steering         # predicts with a steering command older than 0.2 s (cumulative)
