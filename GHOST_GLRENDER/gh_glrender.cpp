
#include "cylinder.h"
#include "gh_glrender.h"

#include <algorithm>

#include <gl\glut.h>


#include <cv_skeleton.h>
#include <cv_pointmat_common.h>

#include <gh_common.h>
#include <gh_search.h>
#include <gh_render.h>
#include <gh_texture.h>

#include <recons_voxel_integration.h>
#include <recons_marchingcubes.h>

#include <fbolib.h>
#include <glcv.h>



#define USE_KINECT_INTRINSICS 1
float ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0;

//loading filepaths
std::string video_directory = "";
std::string voxel_recons_path = "";
std::string extension = ".xml.gz";
std::string packaged_file_path = "";

//frame data for rendering
std::vector<FrameDataProcessed> frame_datas;
//std::vector<FrameData> frame_datas_unprocessed;
std::vector<SkeletonNodeHardMap> snhmaps;
BodypartFrameCluster bodypart_frame_cluster;
std::vector<Cylinder> cylinders;
std::vector<VoxelMatrix> voxels;
float voxel_size;
BodyPartDefinitionVector bpdv;
std::vector<std::vector<cv::Vec3f>> bodypart_precalculated_rotation_vectors;

//opengl vertex information for rendering
std::vector<std::vector<float>> triangle_vertices;
std::vector<std::vector<unsigned int>> triangle_indices;
std::vector<std::vector<unsigned char>> triangle_colors;
std::vector<std::vector<float>> triangle_texcoords;

//parameters for rendering
int numframes = 10;
bool skip_side = false;
float tsdf_offset = 0;

//rendering opengl texture list (for circle and triangle texture mapping)
std::vector<GLuint> bodypart_texture;

//important transforms for rendering
cv::Mat model_center;
cv::Mat model_center_inv;
cv::Mat opengl_projection;// , frame_opengl_projection;

//rendering bg color
cv::Vec3b bg_color;
cv::Scalar output_bg_color;

//opengl stuff
GLUquadric * quadric;
FBO fbo1(1000, 1000);

//frame animation stuff
//int anim_frame = 0;
//float anim_frame_f = 0;
//#define ANIM_DEFAULT_FPS 12

double z_near, z_far;

int glrender_load(int argc, char ** argv, double zNear, double zFar, int * out_width, int * out_height){

	z_near = zNear;
	z_far = zFar;

	if (USE_KINECT_INTRINSICS){
		cv::FileStorage fs;
		fs.open("out_cameramatrix_test.yml", cv::FileStorage::READ);
		fs["alpha"] >> ki_alpha;
		fs["beta"] >> ki_beta;
		fs["gamma"] >> ki_gamma;
		fs["u"] >> ki_u0;
		fs["v"] >> ki_v0;
	}

	packaged_file_path = "";

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
			numframes = atoi(argv[i + 1]);
			++i;
		}
		else if (strcmp(argv[i], "-s") == 0){
			skip_side = true;
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
				<< "-p: packaged file\n";
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
		//std::vector<PointMap> pointmaps;

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

		bodypart_frame_cluster = cluster_frames(64, bpdv, snhmaps, frame_datas, 50);
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
		load_packaged_file(packaged_file_path, bpdv, frame_datas, bodypart_frame_cluster, triangle_vertices, triangle_indices, voxels, voxel_size, cylinders);
		triangle_colors.resize(bpdv.size());
		triangle_texcoords.resize(bpdv.size());
		bodypart_texture.resize(bpdv.size());
#if VIEW_DEPENDENT_TEXTURE == 1
		for (int i = 0; i < bpdv.size(); ++i){
			for (int j = 0; j < triangle_indices[i].size(); ++j){

				triangle_colors[i].push_back(bpdv[i].mColor[0] * 0xff);
				triangle_colors[i].push_back(bpdv[i].mColor[1] * 0xff);
				triangle_colors[i].push_back(bpdv[i].mColor[2] * 0xff);
			}
		}
#else
#endif

		for (int i = 0; i < frame_datas.size(); ++i){
			snhmaps.push_back(SkeletonNodeHardMap());
			cv_draw_and_build_skeleton(&frame_datas[i].mRoot, cv::Mat::eye(4, 4, CV_32F), frame_datas[i].mCameraMatrix, frame_datas[i].mCameraPose, &snhmaps[i]);
		}
	}




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

	glClearColor(0.5f, 0.5f, 0.5f, 1.f);
	bg_color = cv::Vec3b(0.5 * 0xff, 0.5 * 0xff, 0.5 * 0xff);
	output_bg_color = cv::Scalar(0.5 * 0xff, 0.5 * 0xff, 0.5 * 0xff);


	if (out_width != 0){
		*out_width = frame_datas[0].mWidth;
	}
	if (out_height != 0){
		*out_height = frame_datas[0].mHeight;
	}

	return 1;
}

void glrender_init(){


	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);    /* Uses default lighting parameters */

	glEnable(GL_DEPTH_TEST);

	//glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glEnable(GL_NORMALIZE);


	//glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	glutGet(GLUT_ELAPSED_TIME);



	//frame_win_width = win_width;
	//frame_win_height = win_height;
	//frame_opengl_projection = opengl_projection.clone();

	quadric = gluNewQuadric();
	genFBO(fbo1);

	bodypart_precalculated_rotation_vectors.resize(bpdv.size());
	for (int i = 0; i < bpdv.size(); ++i){
		bodypart_precalculated_rotation_vectors[i] = precalculate_vecs(bpdv[i], snhmaps, frame_datas);
	}


	//need to do textures after gl context
#if VIEW_DEPENDENT_TEXTURE != 1 
	if (packaged_file_path != ""){
		if (video_directory == ""){
			std::size_t found = packaged_file_path.find_last_of("/\\");
			video_directory = packaged_file_path.substr(0, found);
		}
		std::stringstream ss;
		glEnable(GL_TEXTURE_2D);
		glGenTextures(bpdv.size(), bodypart_texture.data());

#if TEXTURE_MAP_CYLINDER == 1

		std::string texture_dir = video_directory + "/cylinder_textures/";

		for (int i = 0; i < bpdv.size(); ++i){
			ss.str("");
			ss << texture_dir << "texture" << i << ".png";
			cv::Mat texture = cv::imread(ss.str());

			cv::Mat texture_flip = texture.clone();

			cv::Mat texture_flip_bgr;
			cv::cvtColor(texture_flip, texture_flip_bgr, CV_BGR2RGB);

			mat_to_texture(texture_flip_bgr, bodypart_texture[i]);

			for (int j = 0; j < triangle_vertices[i].size(); j += 3){
				cv::Vec4f vertex(triangle_vertices[i][j], triangle_vertices[i][j + 1], triangle_vertices[i][j + 2], 1);
				float azimuth, height;
				cartesian_to_cylinder_nodist(vertex, azimuth, height);
				int U, V;
				cylinder_to_uv(azimuth, height, voxels[i].height, texture.cols, texture.rows, U, V);

				if (U < 0) U = 0;
				else if (U > texture.cols - 1)U = texture.cols - 1;
				if (V < 0) V = 0;
				else if (V > texture.rows - 1)V = texture.rows - 1;

				cv::Vec3b color = texture.ptr<cv::Vec3b>(V)[U];

				triangle_colors[i].push_back(color[0]);
				triangle_colors[i].push_back(color[1]);
				triangle_colors[i].push_back(color[2]);

				triangle_texcoords[i].push_back(U);
				triangle_texcoords[i].push_back(V);
			}

		}
#elif TEXTURE_MAP_TRIANGLES == 1

		std::string texture_dir = video_directory + "/triangle_textures/";

		cv::FileStorage fs;

		for (int i = 0; i < bpdv.size(); ++i){
			ss.str("");
			ss << texture_dir << "texture" << i << ".png";
			cv::Mat texture = cv::imread(ss.str());

			cv::Mat texture_flip = texture.clone();

			cv::Mat texture_flip_bgr;
			cv::cvtColor(texture_flip, texture_flip_bgr, CV_BGR2RGB);

			mat_to_texture(texture_flip_bgr, bodypart_texture[i]);

			std::vector<cv::Vec3f> new_vertices(triangle_indices[i].size());

			//redo triangle_indices, duplicating the vertices that are shared by multiple triangles
			for (int j = 0; j < triangle_indices[i].size(); j += 3){
				int ind1 = triangle_indices[i][j];
				int ind2 = triangle_indices[i][j + 1];
				int ind3 = triangle_indices[i][j + 2];

				cv::Vec3f vert1(triangle_vertices[i][ind1 * 3], triangle_vertices[i][ind1 * 3 + 1], triangle_vertices[i][ind1 * 3 + 2]);
				cv::Vec3f vert2(triangle_vertices[i][ind2 * 3], triangle_vertices[i][ind2 * 3 + 1], triangle_vertices[i][ind2 * 3 + 2]);
				cv::Vec3f vert3(triangle_vertices[i][ind3 * 3], triangle_vertices[i][ind3 * 3 + 1], triangle_vertices[i][ind3 * 3 + 2]);

				new_vertices[j + 0] = (vert1);
				new_vertices[j + 1] = (vert2);
				new_vertices[j + 2] = (vert3);
			}

			triangle_indices[i].clear();
			triangle_vertices[i].clear();
			triangle_colors[i].clear();
			triangle_texcoords[i].clear();

			triangle_indices[i].resize(new_vertices.size());
			triangle_vertices[i].resize(new_vertices.size() * 3);

			for (int j = 0; j < new_vertices.size(); ++j){
				triangle_indices[i][j] = j;
				triangle_vertices[i][j * 3 + 0] = new_vertices[j](0);
				triangle_vertices[i][j * 3 + 1] = new_vertices[j](1);
				triangle_vertices[i][j * 3 + 2] = new_vertices[j](2);
			}

			ss.str("");
			ss << texture_dir << "/UV" << i << ".xml";
			fs.open(ss.str(), cv::FileStorage::READ);

			cv::FileNode uv_node = fs["UV"];

			for (auto it = uv_node.begin(); it != uv_node.end(); ++it){
				int U1, V1, U2, V2, U3, V3;

				(*it)["U1"] >> U1;
				(*it)["V1"] >> V1;
				(*it)["U2"] >> U2;
				(*it)["V2"] >> V2;
				(*it)["U3"] >> U3;
				(*it)["V3"] >> V3;

				triangle_texcoords[i].push_back(U1);
				triangle_texcoords[i].push_back(V1);
				triangle_texcoords[i].push_back(U2);
				triangle_texcoords[i].push_back(V2);
				triangle_texcoords[i].push_back(U3);
				triangle_texcoords[i].push_back(V3);
			}
		}
#endif
	}
#endif

}

cv::Mat glrender_display(int anim_frame, const cv::Mat& opengl_modelview, int win_width, int win_height, int flags){

	bool debug_shape_cylinders = flags & GLR_SHAPE_CYLINDER;
	bool debug_untextured = flags & GLR_UNTEXTURED;
	bool debug_show_normals = flags & GLR_SHOW_NORMALS;
	bool debug_inspect_texture_map = flags & GLR_INSPECT_TEXTURE_MAP;

	//anim_frame_f += (elapsed_time * ANIM_DEFAULT_FPS / 1000.f);
	//if (anim_frame_f >= snhmaps.size()){
	//	anim_frame_f -= snhmaps.size();
	//}
	//anim_frame = anim_frame_f;
	//anim_frame %= snhmaps.size();
	//while (skip_side && frame_datas[anim_frame].mnFacing != FACING_FRONT && frame_datas[anim_frame].mnFacing != FACING_BACK){
	//	++anim_frame_f;
	//	anim_frame = anim_frame_f;
	//	anim_frame %= snhmaps.size();
	//}
	//anim_frame = anim_frame_f;

	assert(anim_frame < snhmaps.size());

	anim_frame %= snhmaps.size();

	set_projection_matrix(frame_datas[anim_frame].mCameraMatrix, win_width, win_height);

	int current_fbo;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);

	glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo1.fboId);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	cv::Mat flip_x = cv::Mat::eye(4, 4, CV_32F);
	//flip_x.ptr<float>(0)[0] = -1;
	cv::Mat transformation = flip_x * model_center * opengl_modelview * model_center_inv;
	{
		cv::Mat transformation_t = transformation.t();
		glMultMatrixf(transformation_t.ptr<float>());

	}

	glEnableClientState(GL_VERTEX_ARRAY);

#if VIEW_DEPENDENT_TEXTURES != 1
#if TEXTURE_MAP_CYLINDER == 1
	glEnableClientState(GL_COLOR_ARRAY);
	glDisable(GL_TEXTURE_2D);
#elif TEXTURE_MAP_TRIANGLES == 1
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnable(GL_TEXTURE_2D);
#endif
#endif

	//glEnable(GL_COLOR_MATERIAL);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	std::vector<cv::Vec3b> bodypart_color(bpdv.size());

	int start_bp = 0;
	//int bp_inc = debug_shape_cylinders ? 1 : bpdv.size();
	int bp_inc = bpdv.size();
	cv::Mat output_img(win_height, win_width, CV_8UC4, cv::Scalar(0.5 * 0xff, 0.5 * 0xff, 0.5 * 0xff, 0));

	int max_search = MAX_SEARCH;

	glDisable(GL_LIGHTING);

	for (int i = 0; i < bpdv.size(); ++i){
		bodypart_color[i] = cv::Vec3b(bpdv[i].mColor[0] * 0xff, bpdv[i].mColor[1] * 0xff, bpdv[i].mColor[2] * 0xff);
	}

	for (start_bp; start_bp < bpdv.size(); start_bp += bp_inc){
		int end_bp = start_bp + bp_inc;
		for (int i = start_bp; i < end_bp; ++i){

			glPushMatrix();

			glBindTexture(GL_TEXTURE_2D, bodypart_texture[i]);

			glColor3f(1, 1, 1);
			if (debug_shape_cylinders){

				cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose)).t();
				glMultMatrixf(transform_t.ptr<float>());

				glColor3ubv(&(bodypart_color[i][0]));

				renderCylinder(0, 0, 0, 0, voxels[i].height * voxel_size, 0, cylinders[i].width, 16, quadric);

			}
			else{
				cv::Mat transform_t = (get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose) * get_voxel_transform(voxels[i].width, voxels[i].height, voxels[i].depth, voxel_size)).t();
				glMultMatrixf(transform_t.ptr<float>());

				glVertexPointer(3, GL_FLOAT, 0, triangle_vertices[i].data());
#if VIEW_DEPENDENT_TEXTURE != 1
#if TEXTURE_MAP_TRIANGLES == 1
				glTexCoordPointer(2, GL_FLOAT, 0, triangle_texcoords[i].data());
#elif TEXTURE_MAP_CYLINDER == 1
				glColorPointer(3, GL_UNSIGNED_BYTE, 0, triangle_colors[i].data());
#endif
#else
				glColor3ubv(&(bodypart_color[i][0]));
#endif

				glDrawElements(GL_TRIANGLES, triangle_indices[i].size(), GL_UNSIGNED_INT, triangle_indices[i].data());
			}

			glPopMatrix();
		}


		glDisableClientState(GL_VERTEX_ARRAY);
		//glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);



		//now take the different body part colors and map em to the proper textures

		if (!debug_untextured && VIEW_DEPENDENT_TEXTURE){

			glPixelStorei(GL_PACK_ALIGNMENT, 1);

			cv::Mat render_pretexture = gl_read_color(win_width, win_height);

			//shrink by 2 pix
			//cv::Mat black_mask(render_pretexture.size(), CV_8U, cv::Scalar(0));
			//for (int y = 0; y < render_pretexture.rows; ++y){
			//	for (int x = 0; x < render_pretexture.cols; ++x){
			//		if (render_pretexture.ptr<cv::Vec3b>(y)[x] == bg_color){
			//			black_mask.ptr<unsigned char>(y)[x] = 0xff;
			//		}
			//	}
			//}
			//cv::Mat black_mask_dilate;
			//cv::dilate(black_mask, black_mask_dilate, cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5), cv::Point(2, 2)));
			//for (int y = 0; y < render_pretexture.rows; ++y){
			//	for (int x = 0; x < render_pretexture.cols; ++x){
			//		if (black_mask_dilate.ptr<unsigned char>(y)[x] == 0xff){
			//			render_pretexture.ptr<cv::Vec3b>(y)[x] = bg_color;
			//		}
			//	}
			//}

#if DEBUG_OUTPUT_TEXTURE
			//debug
			debug_ss.str("");
			debug_ss << debug_print_dir << "/texture-time" << timestamp << ".png";
			cv::imwrite(debug_ss.str(), render_pretexture);
#endif

			cv::Mat render_depth = gl_read_depth(win_width, win_height, opengl_projection);

			if (debug_show_normals){
				cv::Mat normal_img = draw_normals(render_depth, frame_datas[anim_frame].mCameraMatrix);

				for (int y = 0; y < win_height; ++y){
					for (int x = 0; x < win_width; ++x){
						const cv::Vec3f& normal = normal_img.ptr<cv::Vec3f>(y)[x];
						output_img.ptr<cv::Vec4b>(y)[x] = cv::Vec4b(normal(0) * 0xff, normal(1) * 0xff, normal(2) * 0xff, 0xff);
					}
				}
			}
			else{
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


				for (int i = 0; i < bpdv.size(); ++i){

					if (bodypart_pts_2d_withdepth_v[i].size() == 0) continue;

					//convert the vector into a matrix
					cv::Mat bodypart_pts = pointvec_to_pointmat(bodypart_pts_2d_withdepth_v[i]);

					//now multiply the inverse bodypart transform + the bodypart transform for the best frame
					//oh yeah, look for the best frame
					//this should probably be in a different function, but how do i access it in display...?
					//,maybe just global vars


					cv::Mat source_transform = transformation * get_bodypart_transform(bpdv[i], snhmaps[anim_frame], frame_datas[anim_frame].mCameraPose);

					//cv::Mat flip_x = cv::Mat::eye(4, 4, CV_32F);
					//flip_x.ptr<float>(0)[0] = -1;
					//unsigned int best_frame = find_best_frame(bpdv[i], source_transform, snhmaps, bodypart_frame_cluster[i]);
					std::vector<unsigned int> best_frames = sort_best_frames(bpdv[i], flip_x * source_transform, snhmaps, frame_datas, bodypart_precalculated_rotation_vectors[i], bodypart_frame_cluster[i]);

					cv::Mat neutral_pts = (frame_datas[anim_frame].mCameraMatrix * source_transform).inv() * bodypart_pts;

					int search_limit = std::min((int)best_frames.size(), max_search);

#if DEBUG_OUTPUT_TEXTURE
					std::ofstream of;
					debug_ss.str("");
					debug_ss << debug_print_dir << "/texture-time" << timestamp << "-bp" << i << ".txt";
					of.open(debug_ss.str());
					const cv::Mat& cmp_rot_only = source_transform(cv::Range(0, 3), cv::Range(0, 3));
					cv::Vec3f cmp_rot_vec;
					cv::Rodrigues(cmp_rot_only, cmp_rot_vec);
					of << "input: " << cmp_rot_vec(0) << " " << cmp_rot_vec(1) << " " << cmp_rot_vec(2) << std::endl;

					//debug
					for (int it = 0; it < search_limit; ++it){
						debug_ss.str("");
						debug_ss << debug_print_dir << "/texture-time" << timestamp << "-bp" << i << "-rank" << it << ".png";
						cv::imwrite(debug_ss.str(), frame_datas[best_frames[it]].mBodyPartImages[i].mMat);
						of << "frame " << best_frames[it] << ": " << bodypart_precalculated_rotation_vectors[i][best_frames[it]](0)
							<< " " << bodypart_precalculated_rotation_vectors[i][best_frames[it]](1)
							<< " " << bodypart_precalculated_rotation_vectors[i][best_frames[it]](2)
							<< std::endl;
					}

					of.close();
#endif

					for (int best_frames_it = 0; best_frames_it < search_limit; ++best_frames_it){

						unsigned int best_frame = best_frames[best_frames_it];

						//if (bpdv[i].mBodyPartName == "HEAD"){
						//	std::cout << "head best frame: " << best_frame << "; actual frame: " << anim_frame << std::endl;
						//}
						cv::Mat target_transform = get_bodypart_transform(bpdv[i], snhmaps[best_frame], frame_datas[best_frame].mCameraPose);
						//cv::Mat bodypart_img_uncropped = uncrop_mat(frame_datas[best_frame].mBodyPartImages[i], cv::Vec3b(0xff, 0xff, 0xff)); //uncrop is slow, just offset the cropped mat

						cv::Mat neutral_pts_occluded;
						std::vector<cv::Point2i> _2d_pts_occluded;

						if (debug_shape_cylinders){
							inverse_point_mapping(neutral_pts, bodypart_pts_2d_v[i], frame_datas[best_frame].mCameraMatrix, target_transform,
								frame_datas[best_frame].mBodyImage.mMat, frame_datas[best_frame].mBodyImage.mOffset, output_img, neutral_pts_occluded, _2d_pts_occluded, !debug_shape_cylinders, debug_inspect_texture_map);

						}
						else{
							inverse_point_mapping(neutral_pts, bodypart_pts_2d_v[i], frame_datas[best_frame].mCameraMatrix, target_transform,
								frame_datas[best_frame].mBodyPartImages[i].mMat, frame_datas[best_frame].mBodyPartImages[i].mOffset, output_img, neutral_pts_occluded, _2d_pts_occluded, !debug_shape_cylinders, debug_inspect_texture_map);
						}
						bodypart_pts_2d_v[i] = _2d_pts_occluded;
						if (!_2d_pts_occluded.empty()){
							neutral_pts = neutral_pts_occluded(cv::Range(0, 4), cv::Range(0, _2d_pts_occluded.size()));
						}
						else{
							break;
						}
					}

					if (!bodypart_pts_2d_v[i].empty()){
						bool up = true;

						for (int iter = 0; iter < FILL_LIMIT && !bodypart_pts_2d_v[i].empty(); ++iter){

							for (int _n = 0; _n < bodypart_pts_2d_v[i].size(); ++_n){

								int n = up ? _n : bodypart_pts_2d_v[i].size() - _n - 1;

								int npix = 0;
								int px = bodypart_pts_2d_v[i][n].x;
								int py = bodypart_pts_2d_v[i][n].y;

								int av_b = 0, av_g = 0, av_r = 0;

								for (int fx = -FILL_NEIGHBORHOOD; fx < FILL_NEIGHBORHOOD; ++fx){
									for (int fy = -FILL_NEIGHBORHOOD; fy < FILL_NEIGHBORHOOD; ++fy){
										int wx = px + fx;
										int wy = py + fy;
										if (CLAMP(wx, wy, output_img.cols, output_img.rows)){
											const cv::Vec4b& color = output_img.ptr<cv::Vec4b>(wy)[wx];
											if (color(3) > 0){
												av_b += color(0);
												av_g += color(1);
												av_r += color(2);
												++npix;
											}
										}
									}
								}

								if (npix > 0){
									av_b /= npix;
									av_g /= npix;
									av_r /= npix;

									output_img.ptr<cv::Vec4b>(py)[px] = cv::Vec4b(av_b, av_g, av_r, 0xff);

									bodypart_pts_2d_v[i].erase(bodypart_pts_2d_v[i].begin() + n);
									--n;
								}
							}
							up = !up;
						}
					}
				}
			}


#if DEBUG_OUTPUT_TEXTURE
			debug_ss.str("");
			debug_ss << debug_print_dir << "/texture-time" << timestamp << "-render.png";
			cv::imwrite(debug_ss.str(), output_img_flip);
#endif
		}
		else{

			cv::Mat render_pretexture = gl_read_color(win_width, win_height);

			//cv::Mat render_pretexture_flip;
			//cv::flip(render_pretexture, render_pretexture_flip, 0);
			//display_mat(render_pretexture_flip, true);

			output_img = render_pretexture;
		}
	}

	cv::Mat output_img_flip;
	cv::flip(output_img, output_img_flip, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);

	return output_img_flip;
}

void glrender_skeleton(int anim_frame, const cv::Mat& opengl_modelview){

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

void set_projection_matrix(const cv::Mat& camera_matrix, int win_width, int win_height){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	float ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0;
	ki_alpha = camera_matrix.ptr<float>(0)[0];
	ki_beta = camera_matrix.ptr<float>(1)[1];
	ki_gamma = camera_matrix.ptr<float>(0)[1];
	ki_u0 = camera_matrix.ptr<float>(0)[2];
	ki_v0 = camera_matrix.ptr<float>(1)[2];
	int viewport[4];
	cv::Mat proj_t = build_opengl_projection_for_intrinsics_2(viewport, -ki_alpha, ki_beta, ki_gamma, ki_u0, ki_v0 + 10, win_width, win_height, z_near, z_far).t(); //im not proud of this
	glMultMatrixf(proj_t.ptr<float>());


	glViewport(0, 0, win_width, win_height);

	opengl_projection.create(4, 4, CV_32F);
	glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)opengl_projection.data);
	opengl_projection = opengl_projection.t();

}


void glrender_release(){
	gluDeleteQuadric(quadric);
}

size_t glrender_get_numframes(){
	return snhmaps.size();
}