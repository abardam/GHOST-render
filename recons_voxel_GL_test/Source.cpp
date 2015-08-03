/*
ITP (In This Project) we attempt to perform marching cubes on the reconstructed voxels to generate a mesh, which we will then render using opengl.

*/

#include <stdlib.h>
#include <stdio.h>

#include <GL/glut.h>

#include <map>
#include <vector>
#include <sstream>

#include <opencv2\opencv.hpp>

//#include <recons_common.h>
#define AI_DEG_TO_RAD(x) ((x)*0.0174532925f)

#include <glcv.h>

#include <ctime>

#define NOMINMAX
#include <Windows.h>
#include <fstream>

#include "gh_glrender.h"

float zNear = 0.1, zFar = 10.0;


#define DEBUG_OUTPUT_TEXTURE 0

//manual zoom
float zoom = 1.f;

//mouse
int mouse_x, mouse_y;
bool mouse_down = false;
bool auto_rotate = true;

#define ZOOM_VALUE 0.1
#define ROTATE_VALUE 1

//int frame_win_width, frame_win_height;
//window name
int window1;

float fovy = 45.;

GLint prev_time = 0;
GLint elapsed_time = 0;
GLint prev_fps_time = 0;

//std::vector<TRIANGLE> tris;

//window dimensions
int win_width, win_height;

int frames = 0;

cv::Mat opengl_modelview;
cv::Mat camera_matrix_current;

bool debug_inspect_texture_map = false;
bool debug_draw_texture = true;
bool debug_draw_skeleton = true;
bool playing = true;
bool debug_shape_cylinders = false;
bool debug_show_normals = false;

//frame animation stuff
int anim_frame = 0;
float anim_frame_f = 0;
#define ANIM_DEFAULT_FPS 12


std::string debug_print_dir;
int debug_ki_alpha_shiz = -1;

std::string generate_debug_print_dir(){
	std::time_t time = std::time(nullptr);
	std::tm ltime;
	localtime_s(&ltime, &time);
	std::stringstream ss;

	ss << "debug-RVGT-" << ltime.tm_year << ltime.tm_mon << ltime.tm_mday << ltime.tm_hour << ltime.tm_min;
	return ss.str();
}

/* ---------------------------------------------------------------------------- */
void reshape(int width, int height)
{
	//const double aspectRatio = (float)width / height, fieldOfView = fovy;
	//
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//
	//if (USE_KINECT_INTRINSICS){
	//	int viewport[4];
	//	cv::Mat proj_t = build_opengl_projection_for_intrinsics(viewport, -ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0+10, width, height, zNear, zFar, -1).t(); //note: ki_alpha is negative. NOTE2: the +10 is FUDGE
	//	glMultMatrixf(proj_t.ptr<float>());
	//}
	//else{
	//	gluPerspective(fieldOfView, aspectRatio,
	//		zNear, zFar);  /* Znear and Zfar */
	//}
	glViewport(0, 0, width, height);
	win_width = width;
	win_height = height;

	//opengl_projection.create(4, 4, CV_32F);
	//glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)opengl_projection.data);
	//opengl_projection = opengl_projection.t();
}

/* ---------------------------------------------------------------------------- */
void do_motion(void)
{

	int time = glutGet(GLUT_ELAPSED_TIME);
	//angle += (time - prev_time)*0.01;
	
	if (playing){
		elapsed_time = time - prev_time;
	}
	else{
		elapsed_time = 0;
	}

	prev_time = time;

	frames += 1;
	if ((time - prev_fps_time) > 1000) /* update every seconds */
	{
		int current_fps = frames * 1000 / (time - prev_fps_time);
		printf("%d fps\n", current_fps);
		frames = 0;
		prev_fps_time = time;
	}


	glutPostRedisplay();
}

void mouseFunc(int button, int state, int x, int y){
	switch (button){
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN){
			mouse_x = x;
			mouse_y = y;
			mouse_down = true;
		}
		else if (state == GLUT_UP){
			mouse_down = false;
		}
		break;
	case 3: //scroll up?
		zoom += ZOOM_VALUE;
		if (state == GLUT_UP) return;
		break;
	case 4: //scroll down?
		if (state == GLUT_UP) return;
		zoom -= ZOOM_VALUE;
		break;
	}
}

void mouseMoveFunc(int x, int y){
	if (mouse_down){

		cv::Mat angle_x_rot = cv::Mat::eye(4,4,CV_32F);
		cv::Mat angle_y_rot = cv::Mat::eye(4, 4, CV_32F);

		float angle_x_rad = AI_DEG_TO_RAD((x - mouse_x));
		float angle_y_rad = AI_DEG_TO_RAD((y - mouse_y));

		cv::Rodrigues(cv::Vec3f(0, angle_x_rad, 0), angle_x_rot(cv::Range(0,3),cv::Range(0,3)));
		cv::Rodrigues(cv::Vec3f(angle_y_rad, 0, 0), angle_y_rot(cv::Range(0,3),cv::Range(0,3)));

		opengl_modelview = angle_x_rot * angle_y_rot * opengl_modelview;

		auto_rotate = false;

		mouse_x = x;
		mouse_y = y;
	}
}

void keyboardFunc(unsigned char key, int x, int y){
	key = tolower(key);
	if (key == 'i'){
		debug_inspect_texture_map = !debug_inspect_texture_map;
	}
	else if (key == 't'){
		debug_draw_texture = !debug_draw_texture;
	}
	else if (key == 's'){
		debug_draw_skeleton = !debug_draw_skeleton;
	}
	else if (key == 'p'){
		playing = !playing;
	}
	else if (key == 'c'){
		debug_shape_cylinders = !debug_shape_cylinders;
	}
	else if (key == 'a'){
		debug_ki_alpha_shiz *= -1;
	}
	else if (key == 'n'){
		debug_show_normals = !debug_show_normals;
	}
}


/* ---------------------------------------------------------------------------- */
void display(void)
{
	glutSetWindow(window1);

	float tmp;



	//if (auto_rotate){
	//	gluLookAt(0.f, 0.f, 3.f, 0.f, 0.f, -5.f, 0.f, 1.f, 0.f);
	//	/* rotate it around the y axis */
	//	glRotatef(angle, 0.f, 1.f, 0.f);
	//	glGetFloatv(GL_MODELVIEW_MATRIX, current_matrix);
	//}
	//else{
	//	glMultMatrixf(current_matrix);
	//
	//	glRotatef(angle_x, 0.f, 1.f, 0.f);
	//	glRotatef(angle_y, 1.f, 0.f, 0.f);
	//
	//	glGetFloatv(GL_MODELVIEW_MATRIX, current_matrix);
	//
	//	angle_x = 0;
	//	angle_y = 0;
	//
	//	glScalef(zoom, zoom, zoom);
	//
	//}

	unsigned int timestamp = std::time(nullptr);
	std::stringstream debug_ss;
	debug_ss << debug_print_dir << "/" << "debug" << timestamp << ".txt";
	//std::ofstream output;
	//output.open(debug_ss.str());

	int flags = 0;
	if (!debug_draw_texture) flags |= GLR_UNTEXTURED;
	if (debug_inspect_texture_map) flags |= GLR_INSPECT_TEXTURE_MAP;
	if (debug_shape_cylinders) flags |= GLR_SHAPE_CYLINDER;
	if (debug_show_normals) flags |= GLR_SHOW_NORMALS;

	size_t max_frames = glrender_get_numframes();


	anim_frame_f += (elapsed_time * ANIM_DEFAULT_FPS / 1000.f);
	while (anim_frame_f >= max_frames){
		anim_frame_f -= max_frames;
	}
	anim_frame = anim_frame_f;

	cv::Mat output_img = glrender_display(anim_frame, opengl_modelview, win_width, win_height, flags);

	glClear(GL_COLOR);

	//now display the rendered pts
	display_mat(output_img, true);

	if (debug_draw_skeleton){

		glrender_skeleton(anim_frame, opengl_modelview);
	}
	
	glutSwapBuffers();
	do_motion();

	//debug
	//output.close();


	//frame_win_width = win_width;
	//frame_win_height = win_height;
	//frame_opengl_projection = opengl_projection.clone();
}


/* ---------------------------------------------------------------------------- */
int main(int argc, char **argv)
{

	debug_print_dir = generate_debug_print_dir();
	CreateDirectory(debug_print_dir.c_str(), nullptr);


	opengl_modelview = cv::Mat::eye(4, 4, CV_32F);

	//do load here
	//remember to return win_width and win_height
	glrender_load(argc, argv, zNear, zFar, &win_width, &win_height);

	glutInitWindowSize(win_width, win_height);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInit(&argc, argv);

	window1 = glutCreateWindow(argv[0]);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	glutMouseFunc(mouseFunc);
	glutMotionFunc(mouseMoveFunc);
	glutKeyboardUpFunc(keyboardFunc);

	reshape(win_width, win_height);

	glrender_init();

	glutMainLoop();

	glrender_release();

	return 0;
}
