#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace cv; 
using namespace std;
const double lambda = 0.0001;
const int width = 28;
const int height = 28;
const double e = 2.71828182845904;
const double pi = 3.14159265359;
int ReverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<int>&labels){
	ifstream file;
	file.open(filename.c_str(), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; i++){
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((int)label);
		}	
	}
}

void read_Mnist_Images(string filename, vector<vector<double> >&images){
	ifstream file(filename.c_str(), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < number_of_images; i++){
			vector<double>tp;
			for (int r = 0; r < n_rows; r++){
				for (int c = 0; c < n_cols; c++){
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
}

Mat conver(const vector<double>& img){
	Mat src(28,28,CV_8U,Scalar(0));
    for(int i = 0;i < width; i++)
    for(int j = 0;j < height; j++){
        src.at<uchar>(i, j) = img[i * height + j];
    }
    resize(src,src,Size(300,300));
	return src;
}
double randomDouble(){
    double den = 2147483647;
    return rand()/den;
}
double randNormal(double std){ //the mean value equal to zero;
    double t1 = randomDouble();
    double t2 = randomDouble();
    return std*sqrt(-2.0*log(t1))*cos(2.0*pi*t2);
}
void init_layer_weight(vector<vector<double > >& weight){
	for(int i = 0;i < weight.size();++i){
		for(int j = 0;j < weight[0].size();++j){
			weight[i][j] = randNormal(1.0/double(weight[0].size()));//randNormal(1.0/n)
		}
	}
}
void init_layer_bias(vector<double>& bias){
	for(int i = 0;i < bias.size(); ++i){
		bias[i] = randNormal(1.0);
	}
}
void momentum_vec1(vector<double>& vec,const double coe){
		for(int  i = 0;i < vec.size(); ++i){
			vec[i]*=coe;
	}
}
void momentum_vec2(vector<vector<double> >& vec,const double coe){
	for(int i = 0;i < vec.size(); ++i){
		for(int  j = 0;j < vec[0].size(); ++j){
			vec[i][j]*=coe;
		}
	}
}
double mul_point(const vector<double>& vec1,const vector<double>& vec2){
	double ans = 0;
	for(int i = 0;i < vec1.size(); ++i){
		ans += vec1[i] * vec2[i];
	}
	return ans;
}
void sigmoid_vec(vector<double>& vec){
	for(int i = 0;i < vec.size(); ++i){
		vec[i] = 1.0 / (1.0 + pow(e,-1.0 * vec[i]));
	}
}
void relu(vector<double>& vec){
	for(int i = 0;i < vec.size(); ++i){
		vec[i] = vec[i] > 0 ? vec[i] : 0;
	}
}
void tanh(vector<double>& vec){
	for(int i = 0;i < vec.size(); ++i){
		double left = pow(e,+1.0 * vec[i]);
		double right = pow(e,-1.0 * vec[i]);
		vec[i] = (left -right)/(left + right);
	}
}
void forward(const vector<double>& pre_layer_data,const vector<vector<double > >& layer_weight,\
const vector<double>& bias,vector<double>& next_layer_data){
    int m = pre_layer_data.size();
    int n = next_layer_data.size();
    if(layer_weight.size() != n || bias.size() != n){
		cout << pre_layer_data.size() << " " << layer_weight.size() << " " << bias.size() << " " << next_layer_data.size() << endl;
        cout << "forward: next_layer_data.size() != layer_weight.size() || bia" << endl;
        exit(0);  
    }
    if(layer_weight[0].size() != pre_layer_data.size()){
        cout << "layer_weight[0].size() != pre_layer_data" << endl;
        exit(0);
    }
    for(int i = 0;i < n; ++i){
        next_layer_data[i] = mul_point(pre_layer_data,layer_weight[i]) + bias[i]; // y = wx + b
    }
}

void minus_max(vector<double>& vec){
	double substr = vec[0];
	for (int i = 1; i < vec.size(); ++i){
		substr = max(substr, vec[i]);
	}
	for (int i = 0; i < vec.size(); ++i){
		vec[i] -= substr;
	}
}
void softmax_vec(vector<double>& vec){
	double deno = 0;
	for(int i = 0;i < vec.size(); ++i){
		deno += pow(e,vec[i]);
	}
	for(int i = 0;i < vec.size(); ++i){
		vec[i] = pow(e,vec[i])/deno;
	}
}
void update_vec2(vector<vector<double> >& weight,const vector<vector<double> >& diff,const double lr){
	for(int  i = 0;i < weight.size(); ++i){
		for(int j = 0;j < weight[0].size(); ++j){
			weight[i][j] += -1.0 * diff[i][j]  * lr - lambda * weight[i][j];
		}
	}
}

void update_vec1(vector<double>& weight,const vector<double>& diff,const double lr){
	for(int  i = 0;i < weight.size(); ++i){
		weight[i] += -1.0 * diff[i]  * lr;
	}
}
int find_best(const vector<double>& vec){
	int ans = 0;
	for(int i = 1;i < vec.size(); ++i){
		if(vec[i] > vec[ans])
			ans = i;
	}
	return ans;
}