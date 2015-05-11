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

#include <AssimpOpenGL.h>
#include <recons_common.h>
#include <ReconsVoxel.h>

#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include <glcv.h>
#include <gh_render.h>
#include <gh_search.h>
#include <gh_common.h>

float zNear = 1.0, zFar = 10.0;


//manual zoom
float zoom = 1.f;

//mouse
int mouse_x, mouse_y;
bool mouse_down = false;
bool auto_rotate = true;

#define ZOOM_VALUE 0.1
#define ROTATE_VALUE 1
#define ANIM_DEFAULT_FPS 12

//window dimensions
int win_width, win_height;
//int frame_win_width, frame_win_height;
//window name
int window1;

//bg color
cv::Vec3b bg_color;
cv::Scalar output_bg_color;

float fovy = 45.;

GLint prev_time = 0;
GLint prev_fps_time = 0;
int frames = 0;

//std::vector<TRIANGLE> tris;

std::vector<std::vector<float>> triangle_vertices;
std::vector<std::vector<unsigned int>> triangle_indices;
std::vector<std::vector<unsigned char>> triangle_colors;

BodyPartDefinitionVector bpdv;
std::vector<SkeletonNodeHardMap> snhmaps;
std::vector<Cylinder> cylinders;
std::vector<VoxelMatrix> voxels;
float voxel_size;

std::vector<FrameDataProcessed> frame_datas;
std::vector<FrameData> frame_datas_unprocessed;

cv::Mat opengl_projection;// , frame_opengl_projection;
cv::Mat opengl_modelview;
cv::Mat camera_matrix_current;
cv::Mat model_center;
cv::Mat model_center_inv;

BodypartFrameCluster bodypart_frame_cluster;

bool debug_inspect_texture_map = false;
bool debug_draw_texture = true;
/* ---------------------------------------------------------------------------- */
void reshape(int width, int height)
{
	width = width / 4 * 4;
	const double aspectRatio = (float)width / height, fieldOfView = fovy;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, aspectRatio,
		zNear, zFar);  /* Znear and Zfar */
	glViewport(0, 0, width, height);
	win_width = width;
	win_height = height;

	opengl_projection.create(4, 4, CV_32F);
	glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)opengl_projection.data);
	opengl_projection = opengl_projection.t();

	camera_matrix_current = generate_camera_intrinsic(width, height, fovy);
}

/* ---------------------------------------------------------------------------- */
void do_motion(void)
{

	int time = glutGet(GLUT_ELAPSED_TIME);
	//angle += (time - prev_time)*0.01;
	
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
	if (key == 'i' || key == 'I'){
		debug_inspect_texture_map = !debug_inspect_texture_map;
	}
	if (key == 't' || key == 'T'){
		debug_draw_texture = !debug_draw_texture;
	}
}


/* ---------------------------------------------------------------------------- */
void display(void)
{
	glutSetWindow(window1);

	float tmp;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

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
	cv::Mat transformation = model_center * opengl_modelview * model_center_inv;

	{		
		cv::Mat transformation_t = transformation.t();
		glMultMatrixf(transformation_t.ptr<float>());
	}

	//glBegin(GL_TRIANGLES);
	//
	//for (int i = 0; i < tris.size(); ++i){
	//	glVertex3f(tris[i].p[0](0), tris[i].p[0](1), tris[i].p[0](2));
	//	glVertex3f(tris[i].p[1](0), tris[i].p[1](1), tris[i].p[1](2));
	//	glVertex3f(tris[i].p[2](0), tris[i].p[2](1), tris[i].p[2](2));
	//}
	//
	//glEnd();

	int anim_frame = (prev_time * ANIM_DEFAULT_FPS / 1000) % snhmaps.size();

	glEnableClientState(GL_VERTEX_ARRAY);

	//glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	for (int i = 0; i < bpdv.size(); ++i){
		glPushMatrix();
		cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose) * get_voxel_transform(voxels[i].width, voxels[i].height, voxels[i].depth, voxel_size)).t();
		glMultMatrixf(transform_t.ptr<float>());

		glVertexPointer(3, GL_FLOAT, 0, triangle_vertices[i].data());
		glColorPointer(3, GL_UNSIGNED_BYTE, 0, triangle_colors[i].data());

		glColor3fv(bpdv[i].mColor);

		glDrawElements(GL_TRIANGLES, triangle_indices[i].size(), GL_UNSIGNED_INT, triangle_indices[i].data());

		glPopMatrix();
	}

	glDisableClientState(GL_VERTEX_ARRAY);



	//now take the different body part colors and map em to the proper textures

	if (debug_draw_texture){

		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		cv::Mat render_pretexture = gl_read_color(win_width, win_height);

		cv::Mat render_depth = gl_read_depth(win_width, win_height, opengl_projection);

		std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
		std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());
		for (int y = 0; y < win_height; ++y){
			for (int x = 0; x < win_width; ++x){
				cv::Vec3b orig_color = render_pretexture.ptr<cv::Vec3b>(y)[x];
				if (orig_color == bg_color) continue;
				for (int i = 0; i < bpdv.size(); ++i){
					cv::Vec3b bp_color(bpdv[i].mColor[0] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[2] * 0xff);

					if (orig_color == bp_color
						){
						float depth = render_depth.ptr<float>(y)[x];
						bodypart_pts_2d_withdepth_v[i].push_back(cv::Vec4f(depth*x, depth*y,
							depth, 1));
						bodypart_pts_2d_v[i].push_back(cv::Point2i(x, y));
						break;
					}
				}
			}
		}

		cv::Mat output_img(win_height, win_width, CV_8UC3, output_bg_color);

		for (int i = 0; i < bpdv.size(); ++i){

			if (bodypart_pts_2d_withdepth_v[i].size() == 0) continue;

			//convert the vector into a matrix
			cv::Mat bodypart_pts = pointvec_to_pointmat(bodypart_pts_2d_withdepth_v[i]);

			//now multiply the inverse bodypart transform + the bodypart transform for the best frame
			//oh yeah, look for the best frame
			//this should probably be in a different function, but how do i access it in display...?
			//,maybe just global vars


			cv::Mat source_transform = transformation * get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose);

			//unsigned int best_frame = find_best_frame(bpdv[i], source_transform, snhmaps, bodypart_frame_cluster[i]);
			std::vector<unsigned int> best_frames = sort_best_frames(bpdv[i], source_transform, snhmaps, frame_datas, bodypart_frame_cluster[i]);


			cv::Mat neutral_pts = (camera_matrix_current * source_transform).inv() * bodypart_pts;

			for (int best_frames_it = 0; best_frames_it < best_frames.size() && !neutral_pts.empty(); ++best_frames_it){

				unsigned int best_frame = best_frames[best_frames_it];

				//if (bpdv[i].mBodyPartName == "HEAD"){
				//	std::cout << "head best frame: " << best_frame << "; actual frame: " << anim_frame << std::endl;
				//}
				cv::Mat target_transform = get_bodypart_transform(bpdv[i], snhmaps[best_frame], frame_datas[best_frame].mCameraPose);
				cv::Mat bodypart_img_uncropped = uncrop_mat(frame_datas[best_frame].mBodyPartImages[i], cv::Vec3b(0xff, 0xff, 0xff));

				cv::Mat neutral_pts_occluded;
				std::vector<cv::Point2i> _2d_pts_occluded;

				inverse_point_mapping(neutral_pts, bodypart_pts_2d_v[i], frame_datas[best_frame].mCameraMatrix, target_transform,
					bodypart_img_uncropped, output_img, neutral_pts_occluded, _2d_pts_occluded, debug_inspect_texture_map);

				neutral_pts = neutral_pts_occluded;
				bodypart_pts_2d_v[i] = _2d_pts_occluded;
			}
		}

		cv::Mat output_img_flip;
		cv::flip(output_img, output_img_flip, 0);

		//now display the rendered pts
		display_mat(output_img_flip, true);
	}
	
	glutSwapBuffers();
	do_motion();


	//frame_win_width = win_width;
	//frame_win_height = win_height;
	//frame_opengl_projection = opengl_projection.clone();
}


/* ---------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
	if (argc <= 2){
		printf("Please enter directory and voxel reconstruct file\n");
		return 0;
	}

	std::string video_directory(argv[1]);
	std::string voxel_recons_path(argv[2]);

	std::stringstream filenameSS;
	int startframe = 0;
	int numframes;
	if (argc >= 3)
	{
		numframes = atoi(argv[3]);
	}
	else{
		numframes = 10;
	}
	cv::FileStorage fs;

	filenameSS << video_directory << "/bodypartdefinitions.xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::READ);
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();
	std::vector<std::string> filenames;

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;

	load_processed_frames(filenames, bpdv.size(), frame_datas);


	std::vector<PointMap> pointmaps;

	//load_frames(filenames, pointmaps, frame_datas_unprocessed);

	for (int i = 0; i < frame_datas.size(); ++i){
		snhmaps.push_back(SkeletonNodeHardMap());
		cv_draw_and_build_skeleton(&frame_datas[i].mRoot, cv::Mat::eye(4,4,CV_32F), frame_datas[i].mCameraMatrix, frame_datas[i].mCameraPose, &snhmaps[i]);
	}

	cv::Vec4f center_pt(0,0,0,0);

	for (int i = 0; i < bpdv.size(); ++i){
		cv::Mat bp_pt_m = get_bodypart_transform(bpdv[i], snhmaps[0], frame_datas[0].mCameraPose)(cv::Range(0, 4), cv::Range(3, 4));
		cv::Vec4f bp_pt = bp_pt_m;
		center_pt += bp_pt;
	}

	center_pt /= center_pt(3);

	model_center = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat(center_pt).copyTo(model_center(cv::Range(0, 4), cv::Range(3, 4)));
	model_center_inv = model_center.inv();

	//filenameSS.str("");
	//filenameSS << video_directory << "/clusters.xml.gz";
	//
	//fs.open(filenameSS.str(), cv::FileStorage::READ);
	//
	//read(fs["bodypart_frame_clusters"], bodypart_frame_cluster);
	//
	//fs.release();

	//bodypart_frame_cluster = cluster_frames(64, bpdv, snhmaps, frame_datas, 1000);
	bodypart_frame_cluster.resize(bpdv.size());

	load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);

	triangle_vertices.resize(bpdv.size());
	triangle_indices.resize(bpdv.size());
	triangle_colors.resize(bpdv.size());

	double num_vertices = 0;

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<TRIANGLE> tri_add;
		
		if (TSDF_array[i].empty()){
			tri_add = marchingcubes_bodypart(voxels[i], voxel_size);
		}
		else{
			tri_add = marchingcubes_bodypart(voxels[i], TSDF_array[i], voxel_size);
		}
		std::vector<cv::Vec4f> vertices;
		std::vector<unsigned int> vertex_indices;
		for (int j = 0; j < tri_add.size(); ++j){
			for (int k = 0; k < 3; ++k){
				cv::Vec4f candidate_vertex = tri_add[j].p[k];
				
				bool vertices_contains_vertex = false;
				int vertices_index;
				for (int l = 0; l < vertices.size(); ++l){
					if (vertices[l] == candidate_vertex){
						vertices_contains_vertex = true;
						vertices_index = l;
						break;
					}
				}
				if (!vertices_contains_vertex){
					vertices.push_back(candidate_vertex);
					vertices_index = vertices.size() - 1;
				}
				vertex_indices.push_back(vertices_index);
			}
		}
		triangle_vertices[i].reserve(vertices.size() * 3);
		triangle_colors[i].reserve(vertices.size() * 3);
		triangle_indices[i].reserve(vertex_indices.size());
		for (int j = 0; j < vertices.size(); ++j){
			triangle_vertices[i].push_back(vertices[j](0));
			triangle_vertices[i].push_back(vertices[j](1));
			triangle_vertices[i].push_back(vertices[j](2));
			triangle_colors[i].push_back(bpdv[i].mColor[0]*255);
			triangle_colors[i].push_back(bpdv[i].mColor[1]*255);
			triangle_colors[i].push_back(bpdv[i].mColor[2]*255);
		}
		num_vertices += vertices.size();
		for (int j = 0; j < vertex_indices.size(); ++j){
			triangle_indices[i].push_back(vertex_indices[j]);
		}
	}

	win_width = frame_datas[0].mWidth;
	win_height = frame_datas[0].mHeight;

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


	glClearColor(0.1f, 0.3f, 0.3f, 1.f);
	bg_color = cv::Vec3b(0.1 * 0xff, 0.1 * 0xff, 0.1 * 0xff);
	output_bg_color = cv::Scalar(0.1 * 0xff, 0.4 * 0xff, 0.3 * 0xff);

	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);    /* Uses default lighting parameters */

	glEnable(GL_DEPTH_TEST);

	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);


	//glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	glutGet(GLUT_ELAPSED_TIME);

	opengl_modelview = cv::Mat::eye(4, 4, CV_32F);

	reshape(win_width, win_height);

	//frame_win_width = win_width;
	//frame_win_height = win_height;
	//frame_opengl_projection = opengl_projection.clone();

	glutMainLoop();

	return 0;
}
