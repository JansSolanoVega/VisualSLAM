from stereo_vo.vo_stereo import *
from utils import *

vo = visual_odometry_stereo(sequence_id=0)
plotter = plotter()
for i in range(get_number_images(vo.img_l_path)):
    vo.process_frame()

    mono_coord = vo.get_mono_coordinates()
    true_coord = vo.get_true_coordinates()

    draw_x, draw_y, draw_z = [int(x) for x in mono_coord]
    true_x, true_y, true_z = [int(x) for x in true_coord]

    traj = plotter.get_trajectory_step(true_x, true_z, draw_x, draw_z)
    plotter.plot_step(traj, vo.curr_frame["l"])
    print("MSE error:", vo.get_mse_error())
cv2.waitKey(0)
cv2.destroyAllWindows()
