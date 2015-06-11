/*
ITP (In This Project) we attempt to perform marching cubes on the reconstructed voxels to generate a mesh, which we will then render using opengl.

*/
#include "cylinder.h"

#include <stdlib.h>
#include <stdio.h>

#include <GL/glut.h>

#include <map>
#include <vector>
#include <sstream>

#include <opencv2\opencv.hpp>

#include <recons_common.h>
#include <ReconsVoxel.h>

#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#include <glcv.h>
#include <gh_render.h>
#include <gh_search.h>
#include <gh_common.h>

#include <fbolib.h>

#include <ctime>

#define NOMINMAX
#include <Windows.h>
#include <fstream>

float zNear = 0.1, zFar = 10.0;

#define MAX_SEARCH 3

#define USE_KINECT_INTRINSICS 1
float ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0;

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
GLint elapsed_time = 0;
GLint prev_fps_time = 0;
int frames = 0;
int anim_frame = 0;
float anim_frame_f = 0;

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
bool debug_draw_skeleton = true;
bool playing = true;
bool debug_shape_cylinders = false;

std::string video_directory = "";
std::string voxel_recons_path = "";
std::string extension = ".xml.gz";
int numframes = 10;
bool skip_side = false;
float tsdf_offset = 0;

GLUquadric * quadric;
FBO fbo1(1000,1000);

std::vector<std::vector<cv::Vec3f>> bodypart_precalculated_rotation_vectors;

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



void load_packaged_file(std::string filename,
	BodyPartDefinitionVector& bpdv,
	std::vector<FrameDataProcessed>& frame_datas,
	BodypartFrameCluster& bodypart_frame_cluster,
	std::vector<std::vector<float>>& triangle_vertices,
	std::vector<std::vector<unsigned int>>& triangle_indices,
	std::vector<VoxelMatrix>& voxels, float& voxel_size){

	int win_width, win_height;

	cv::FileStorage savefile;
	savefile.open(filename, cv::FileStorage::READ);

	cv::FileNode bpdNode = savefile["bodypartdefinitions"];
	bpdv.clear();
	for (auto it = bpdNode.begin(); it != bpdNode.end(); ++it)
	{
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}

	cv::FileNode frameNode = savefile["frame_datas"];
	frame_datas.clear();
	for (auto it = frameNode.begin(); it != frameNode.end(); ++it){
		cv::Mat camera_pose, camera_matrix;
		SkeletonNodeHard root;
		int facing;
		(*it)["camera_extrinsic"] >> camera_pose;
		(*it)["camera_intrinsic_mat"] >> camera_matrix;
		(*it)["skeleton"] >> root;
		(*it)["facing"] >> facing;
		FrameDataProcessed frame_data(bpdv.size(), 0, 0, camera_matrix, camera_pose, root);
		frame_data.mnFacing = facing;
		frame_datas.push_back(frame_data);
	}

	cv::FileNode clusterNode = savefile["bodypart_frame_cluster"];
	bodypart_frame_cluster.clear();
	bodypart_frame_cluster.resize(bpdv.size());
	for (auto it = clusterNode.begin(); it != clusterNode.end(); ++it){
		int bodypart;
		(*it)["bodypart"] >> bodypart;
		cv::FileNode clusterClusterNode = (*it)["clusters"];
		for (auto it2 = clusterClusterNode.begin(); it2 != clusterClusterNode.end(); ++it2){
			int main_frame;
			(*it2)["main_frame"] >> main_frame;
			CroppedMat image;
			(*it2)["image"] >> image;

			std::vector<int> cluster;
			cluster.push_back(main_frame);
			bodypart_frame_cluster[bodypart].push_back(cluster);

			frame_datas[main_frame].mBodyPartImages.resize(bpdv.size());
			frame_datas[main_frame].mBodyPartImages[bodypart] = image;
			win_width = image.mSize.width;
			win_height = image.mSize.height;
		}
	}

	cv::FileNode vertNode = savefile["triangle_vertices"];
	triangle_vertices.clear();
	for (auto it = vertNode.begin(); it != vertNode.end(); ++it){
		triangle_vertices.push_back(std::vector<float>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			float vert;
			(*it2) >> vert;
			triangle_vertices.back().push_back(vert);
		}
	}


	cv::FileNode indNode = savefile["triangle_indices"];
	triangle_indices.clear();
	for (auto it = indNode.begin(); it != indNode.end(); ++it){
		triangle_indices.push_back(std::vector<unsigned int>());
		for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2){
			int ind;
			(*it2) >> ind;
			triangle_indices.back().push_back(ind);
		}
	}

	cv::FileNode voxNode = savefile["voxels"];
	voxels.clear();
	for (auto it = voxNode.begin(); it != voxNode.end(); ++it){
		int width, height, depth;
		(*it)["width"] >> width;
		(*it)["height"] >> height;
		(*it)["depth"] >> depth;
		voxels.push_back(VoxelMatrix(width, height, depth));
	}

	savefile["voxel_size"] >> voxel_size;

	savefile.release();

	frame_datas[0].mWidth = win_width;
	frame_datas[0].mHeight = win_height;
}



/* ---------------------------------------------------------------------------- */
void reshape(int width, int height)
{
	const double aspectRatio = (float)width / height, fieldOfView = fovy;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	if (USE_KINECT_INTRINSICS){
		int viewport[4];
		cv::Mat flip = cv::Mat::eye(4, 4, CV_32F);
		//flip.ptr<float>(2)[2] = -1;
		cv::Mat proj = flip * build_opengl_projection_for_intrinsics(viewport, debug_ki_alpha_shiz * ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0+10, width, height, zNear, zFar, -1); //NOTE: ki_alpha is negative(for some reason openGL switches it). NOTE2: 10 is FUDGE
		
		cv::Mat proj_t = proj.t();

		glMultMatrixf(proj_t.ptr<float>());

		camera_matrix_current = cv::Mat::eye(4, 4, CV_32F);
		camera_matrix_current.ptr<float>(0)[0] = ki_alpha;
		camera_matrix_current.ptr<float>(1)[1] = ki_beta;
		camera_matrix_current.ptr<float>(0)[1] = ki_gamma;
		camera_matrix_current.ptr<float>(0)[2] = ki_u0;
		camera_matrix_current.ptr<float>(1)[2] = ki_v0;

		//debug
		static int projn = 0;
		std::stringstream debug_ss;
		debug_ss << debug_print_dir << "/projection" << projn++ << ".txt";
		std::ofstream output(debug_ss.str());
		output << "projection\n" << proj << std::endl;
		output << "camera_matrix_current\n" << camera_matrix_current << std::endl;
		output.close();
	}
	else{
		gluPerspective(fieldOfView, aspectRatio,
			zNear, zFar);  /* Znear and Zfar */
		camera_matrix_current = generate_camera_intrinsic(width, height, fovy);
	}
	glViewport(0, 0, width, height);
	win_width = width;
	win_height = height;

	opengl_projection.create(4, 4, CV_32F);
	glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)opengl_projection.data);
	opengl_projection = opengl_projection.t();
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
	std::ofstream output;
	output.open(debug_ss.str());

	anim_frame_f += (elapsed_time * ANIM_DEFAULT_FPS / 1000.f);
	if (anim_frame_f >= snhmaps.size()){
		anim_frame_f -= snhmaps.size();
	}
	anim_frame = anim_frame_f;
	anim_frame %= snhmaps.size();
	while (skip_side && frame_datas[anim_frame].mnFacing != FACING_FRONT && frame_datas[anim_frame].mnFacing != FACING_BACK){
		++anim_frame_f; 
		anim_frame = anim_frame_f;
		anim_frame %= snhmaps.size();
	}
	anim_frame = anim_frame_f;


	anim_frame %= snhmaps.size();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo1.fboId);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	cv::Mat transformation = model_center * opengl_modelview * model_center_inv;
	{
		cv::Mat transformation_t = transformation.t();
		glMultMatrixf(transformation_t.ptr<float>());

		//debug
		output << "transformation\n" << transformation << std::endl;
	}

	glEnableClientState(GL_VERTEX_ARRAY);

	//glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//debug
	output << "bodypart_transform\n" << "[\n";

	std::vector<cv::Vec3b> bodypart_color(bpdv.size());

	for (int i = 0; i < bpdv.size(); ++i){
		bodypart_color[i] = cv::Vec3b(bpdv[i].mColor[0] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[2] * 0xff);

		glPushMatrix();

		if (debug_shape_cylinders){

			cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose)).t();
			glMultMatrixf(transform_t.ptr<float>());

			glColor3ubv(&(bodypart_color[i][0]));

			renderCylinder(0, 0, 0, 0, voxels[i].height * voxel_size, 0, cylinders[i].width, 16, quadric);

		}
		else{
			cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose) * get_voxel_transform(voxels[i].width, voxels[i].height, voxels[i].depth, voxel_size)).t();
			glMultMatrixf(transform_t.ptr<float>());

			//debug
			output << transform_t.t() << "\n";

			glVertexPointer(3, GL_FLOAT, 0, triangle_vertices[i].data());
			glColorPointer(3, GL_UNSIGNED_BYTE, 0, triangle_colors[i].data());
			glColor3ubv(&(bodypart_color[i][0]));

			glDrawElements(GL_TRIANGLES, triangle_indices[i].size(), GL_UNSIGNED_INT, triangle_indices[i].data());
		}

		glPopMatrix();
	}

	//debug
	output << "]\n";

	glDisableClientState(GL_VERTEX_ARRAY);



	//now take the different body part colors and map em to the proper textures

	if (debug_draw_texture){

		glPixelStorei(GL_PACK_ALIGNMENT, 1);

		cv::Mat render_pretexture = gl_read_color(win_width, win_height);
		//cv::imwrite("renpre.png", render_pretexture);

		cv::Mat render_depth = gl_read_depth(win_width, win_height, opengl_projection);

		std::vector<std::vector<cv::Vec4f>> bodypart_pts_2d_withdepth_v(bpdv.size());
		std::vector<std::vector<cv::Point2i>> bodypart_pts_2d_v(bpdv.size());
		for (int y = 0; y < win_height; ++y){
			for (int x = 0; x < win_width; ++x){
				cv::Vec3b& orig_color = render_pretexture.ptr<cv::Vec3b>(y)[x];
				if (orig_color == bg_color) continue;
				for (int i = 0; i < bpdv.size(); ++i){
					//cv::Vec3b bp_color(bpdv[i].mColor[0] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[2] * 0xff);

					if (orig_color(0) == (unsigned char)(bodypart_color[i][0]) &&
						orig_color(1) == (unsigned char)(bodypart_color[i][1]) &&
						orig_color(2) == (unsigned char)(bodypart_color[i][2])
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

		cv::Mat output_img(win_height, win_width, CV_8UC4, cv::Scalar(0,0,0,0));

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
			std::vector<unsigned int> best_frames = sort_best_frames(bpdv[i], source_transform, snhmaps, frame_datas, bodypart_precalculated_rotation_vectors[i], bodypart_frame_cluster[i]);


			cv::Mat neutral_pts = (camera_matrix_current * source_transform).inv() * bodypart_pts;

			int search_limit = std::min((int)best_frames.size(), MAX_SEARCH);

			for (int best_frames_it = 0; best_frames_it < best_frames.size() && !neutral_pts.empty(); ++best_frames_it){

				unsigned int best_frame = best_frames[best_frames_it];

				//if (bpdv[i].mBodyPartName == "HEAD"){
				//	std::cout << "head best frame: " << best_frame << "; actual frame: " << anim_frame << std::endl;
				//}
				cv::Mat target_transform = get_bodypart_transform(bpdv[i], snhmaps[best_frame], frame_datas[best_frame].mCameraPose);
				//cv::Mat bodypart_img_uncropped = uncrop_mat(frame_datas[best_frame].mBodyPartImages[i], cv::Vec3b(0xff, 0xff, 0xff)); //uncrop is slow, just offset the cropped mat

				cv::Mat neutral_pts_occluded;
				std::vector<cv::Point2i> _2d_pts_occluded;

				inverse_point_mapping(neutral_pts, bodypart_pts_2d_v[i], frame_datas[best_frame].mCameraMatrix, target_transform,
					frame_datas[best_frame].mBodyPartImages[i].mMat, frame_datas[best_frame].mBodyPartImages[i].mOffset, output_img, neutral_pts_occluded, _2d_pts_occluded, debug_inspect_texture_map);

				neutral_pts = neutral_pts_occluded;
				bodypart_pts_2d_v[i] = _2d_pts_occluded;
			}
		}

		cv::Mat output_img_flip;
		cv::flip(output_img, output_img_flip, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		//now display the rendered pts
		display_mat(output_img_flip, true);
	}
	else{

		cv::Mat render_pretexture = gl_read_color(win_width, win_height);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		cv::Mat render_pretexture_flip;
		cv::flip(render_pretexture, render_pretexture_flip, 0);
		display_mat(render_pretexture_flip, true);
	}

	if (debug_draw_skeleton){

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		cv::Mat transformation = model_center * opengl_modelview * model_center_inv;
		{
			cv::Mat transformation_t = transformation.t();
			glMultMatrixf(transformation_t.ptr<float>());
		}

		glDisable(GL_DEPTH_TEST);
		glColor3f(1.f, 0., 0.);
		glBegin(GL_LINES);
		for (int i = 0; i < bpdv.size(); ++i){
			cv::Mat endpts(4, 2, CV_32F, cv::Scalar(1));
			endpts.ptr<float>(0)[0] = 0;
			endpts.ptr<float>(1)[0] = 0;
			endpts.ptr<float>(2)[0] = 0;
			endpts.ptr<float>(0)[1] = 0;
			endpts.ptr<float>(1)[1] = voxels[i].height * voxel_size;
			endpts.ptr<float>(2)[1] = 0;

			endpts = get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose) * endpts;
			glVertex3f(endpts.ptr<float>(0)[0], endpts.ptr<float>(1)[0], endpts.ptr<float>(2)[0]);
			glVertex3f(endpts.ptr<float>(0)[1], endpts.ptr<float>(1)[1], endpts.ptr<float>(2)[1]);
		}
		glEnd();
		glEnable(GL_DEPTH_TEST);
	}
	
	glutSwapBuffers();
	do_motion();

	//debug
	output.close();


	//frame_win_width = win_width;
	//frame_win_height = win_height;
	//frame_opengl_projection = opengl_projection.clone();
}


/* ---------------------------------------------------------------------------- */
int main(int argc, char **argv)
{

	debug_print_dir = generate_debug_print_dir();
	CreateDirectory(debug_print_dir.c_str(), nullptr);

	if (USE_KINECT_INTRINSICS){
		cv::FileStorage fs;
		fs.open("out_cameramatrix_test.yml", cv::FileStorage::READ);
		fs["alpha"] >> ki_alpha;
		fs["beta"] >> ki_beta;
		fs["gamma"] >> ki_gamma;
		fs["u"] >> ki_u0;
		fs["v"] >> ki_v0;
	}

	std::string packaged_file_path = "";

	for (int i = 1; i < argc; ++i){
		if (strcmp(argv[i], "-d") == 0){
			video_directory = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-v") == 0){
			voxel_recons_path = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-n") == 0){
			numframes = atoi(argv[i+1]);
			++i;
		}
		else if (strcmp(argv[i], "-s") == 0){
			skip_side = true;
			++i;
		}
		else if (strcmp(argv[i], "-t") == 0){
			tsdf_offset = atof(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-e") == 0){
			extension = std::string(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-p") == 0){
			packaged_file_path = std::string(argv[i + 1]);
			++i;
		}
		else{
			std::cout << "Options: -d [video directory] -v [voxel path] -n [num frames] -t [tsdf offset] -e [extension]\n"
				<< "-s: skip non-front and back frames\n"
				<<"-p: packaged file\n";
			return 0;
		}
	}

	if (packaged_file_path == "")
	{
		
		if (video_directory == ""){
			std::cout << "Specify video directory!\n";
			return 0;
		}
		
		if (voxel_recons_path == ""){
			std::cout << "Specify voxel path!\n";
			return 0;
		}
		
		std::stringstream filenameSS;
		int startframe = 0;
		
		cv::FileStorage fs;
		
		filenameSS << video_directory << "/bodypartdefinitions" << extension;
		
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
			filenameSS << video_directory << "/" << frame << extension;
		
			filenames.push_back(filenameSS.str());
		
		}
		
		std::vector<cv::Mat> TSDF_array;
		std::vector<cv::Mat> weight_array;
		
		load_processed_frames(filenames, extension, bpdv.size(), frame_datas, false);
		std::vector<PointMap> pointmaps;
		
		//load_frames(filenames, pointmaps, frame_datas_unprocessed);
		
		for (int i = 0; i < frame_datas.size(); ++i){
			snhmaps.push_back(SkeletonNodeHardMap());
			cv_draw_and_build_skeleton(&frame_datas[i].mRoot, cv::Mat::eye(4, 4, CV_32F), frame_datas[i].mCameraMatrix, frame_datas[i].mCameraPose, &snhmaps[i]);
		}
		
		
		//filenameSS.str("");
		//filenameSS << video_directory << "/clusters.xml.gz";
		//
		//fs.open(filenameSS.str(), cv::FileStorage::READ);
		//
		//read(fs["bodypart_frame_clusters"], bodypart_frame_cluster);
		//
		//fs.release();
		
		bodypart_frame_cluster = cluster_frames(64, bpdv, snhmaps, frame_datas, 1000);
		//bodypart_frame_cluster.resize(bpdv.size());
		//bodypart_frame_cluster = cluster_frames_keyframes(15, bpdv, snhmaps, frame_datas);
		
		load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);
		
		triangle_vertices.resize(bpdv.size());
		triangle_indices.resize(bpdv.size());
		triangle_colors.resize(bpdv.size());
		
		double num_vertices = 0;
		
		for (int i = 0; i < bpdv.size(); ++i){
			std::vector<TRIANGLE> tri_add;
		
			cv::add(tsdf_offset * cv::Mat::ones(TSDF_array[i].rows, TSDF_array[i].cols, CV_32F), TSDF_array[i], TSDF_array[i]);
		
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
				triangle_colors[i].push_back(bpdv[i].mColor[0] * 255);
				triangle_colors[i].push_back(bpdv[i].mColor[1] * 255);
				triangle_colors[i].push_back(bpdv[i].mColor[2] * 255);
			}
			num_vertices += vertices.size();
			for (int j = 0; j < vertex_indices.size(); ++j){
				triangle_indices[i].push_back(vertex_indices[j]);
			}
		}
	}
	else
	{
		load_packaged_file(packaged_file_path, bpdv, frame_datas, bodypart_frame_cluster, triangle_vertices, triangle_indices, voxels, voxel_size);
		triangle_colors.resize(bpdv.size());
		for (int i = 0; i < bpdv.size(); ++i){
			for (int j = 0; j < triangle_indices[i].size(); ++j){
				triangle_colors[i].push_back(bpdv[i].mColor[0] * 0xff);
			}
		}
		for (int i = 0; i < frame_datas.size(); ++i){
			snhmaps.push_back(SkeletonNodeHardMap());
			cv_draw_and_build_skeleton(&frame_datas[i].mRoot, cv::Mat::eye(4, 4, CV_32F), frame_datas[i].mCameraMatrix, frame_datas[i].mCameraPose, &snhmaps[i]);
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

	{
		cv::Vec4f center_pt(0, 0, 0, 0);

		for (int i = 0; i < bpdv.size(); ++i){
			cv::Mat bp_pt_m = get_bodypart_transform(bpdv[i], snhmaps[0], frame_datas[0].mCameraPose)(cv::Range(0, 4), cv::Range(3, 4));
			cv::Vec4f bp_pt = bp_pt_m;
			center_pt += bp_pt;
		}

		center_pt /= center_pt(3);

		model_center = cv::Mat::eye(4, 4, CV_32F);
		cv::Mat(center_pt).copyTo(model_center(cv::Range(0, 4), cv::Range(3, 4)));
		model_center_inv = model_center.inv();
	}

	glClearColor(0.1f, 0.1f, 0.1f, 1.f);
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

	quadric = gluNewQuadric();
	genFBO(fbo1);

	bodypart_precalculated_rotation_vectors.resize(bpdv.size());
	for (int i = 0; i < bpdv.size(); ++i){
		bodypart_precalculated_rotation_vectors[i] = precalculate_vecs(bpdv[i], snhmaps, frame_datas);
	}

	glutMainLoop();

	gluDeleteQuadric(quadric);

	return 0;
}
