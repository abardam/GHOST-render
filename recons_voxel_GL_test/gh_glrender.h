
#include <opencv2\opencv.hpp>


#define GLR_SHAPE_CYLINDER 0x1
#define GLR_UNTEXTURED 0x2
#define GLR_SHOW_NORMALS 0x4
#define GLR_INSPECT_TEXTURE_MAP 0x8

#define VIEW_DEPENDENT_TEXTURE 1
#define TEXTURE_MAP_CYLINDER 0
#define TEXTURE_MAP_TRIANGLES 0

#define MAX_SEARCH 16
#define FILL_LIMIT 16
#define FILL_NEIGHBORHOOD 5

void set_projection_matrix(const cv::Mat& camera_matrix, int win_width, int win_height);

int glrender_load(int argc, char ** argv, double zNear, double zFar, int * out_width, int * out_height);
void glrender_init();
cv::Mat glrender_display(double elapsed_time, const cv::Mat& opengl_modelview, int win_width, int win_height, int flags);
void glrender_release();