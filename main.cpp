#include<stdio.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<string>
#include<algorithm>
#include<random>

// opencv header
#include "opencv2\highgui.hpp"
#include "opencv2\videoio.hpp"
#include "opencv2\opencv.hpp"

// m sequence
#include "mls.h"

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;

// png image to jpg with watermark & key
cv::Mat Encoder(cv::Mat, string, int);
cv::Mat array2block(vector<int>, int, int);

// jpg image to png seperating image and watermark with key
pair<cv::Mat, string> Decoder(cv::Mat, int);
vector<int> block2array(cv::Mat, int, int);

string sync_temp = "1 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 1 0 1 1 0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 0 1 1 1 0 0 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 0 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 1 0 1 1 1 1";
string mes_temp = "0 1 0 0 0 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 0 1 1 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 1 0 1 1 0 0 1 0 1 1 1 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 0 1 1 0 0 0 0 1 1 0 1 0 1 1 0 1 1 1 0 1 0 0 0 1 0 1 0 1 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 1 1 1 0 0 1 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 1 1 0 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 0 1 1 0 1 1 0 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 1 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 1 1 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 1 0 1 0 0 1 1 0 1 0 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 1 0 0 1 1 1 0 1 1 0 0 1 0 1 0 1 1 1 0 1 1 1 1 0 1 0 1 0 0 0 1 1 1 1 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 1 1 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0 1 0 1 0 1 0 0 1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 1 1 0 1 0 0 1 1 1 0 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 0 1 1 0 0 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 1 0 1 1 0 1 1 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 1 0 1 1 0 1 0 0 1 0 1 1 1 0 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 1 1 0 0 0 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 1 1 0 0 1 0 0 1 1 1 0 0 1 0 1 1 0 0 1";

// m-sequence for sync, message
vector<int> sync_seq;
vector<int> message_seq;

map<int, pair<int, int>> match;

int main() {

	string watermark;
	int key = 0;
	cv::Mat img;
	img = cv::imread("1.png"); // image name

	// Watermark and key value setting
	while (watermark.length() != 12) {
		printf("Enter 12 letters watermark :");
		cin >> watermark;
	}

	printf("Enter key value : ");
	cin >> key;
	
	// m-sequenc for sync and message
	for (int i = 0; i < sync_temp.length(); i++) {
		if (sync_temp[i] == ' ')
			continue;
		message_seq.push_back(sync_temp[i] == '1' ? -1 : 1);
	}

	for (int i = 0; i < mes_temp.length(); i++) {
		if (mes_temp[i] == ' ')
			continue;
		sync_seq.push_back(mes_temp[i] == '1' ? -1 : 1);
	}

	cv::imshow("original", img);

	// Encoding
	cv::Mat img_encoded;
	img_encoded = Encoder(img, watermark, key);

	cv::imshow("watermarked img", img_encoded);

	//cv::imwrite("makred_img.jpg, img_encoded);

	// Decoding
	pair <cv::Mat, string> temp;
	cv::Mat img_decoded;
	string wm_decoded;
	//temp = Decoder(img_encoded, key);
	//img_decoded = temp.first;
	//wm_decoded = temp.second;

	//cv::imshow("Decoded img", img_decoded);
	//printf("Decoded Watermark Message : %s\n",wm_decoded);
	
	while (1) {
		char key = (char)cv::waitKey(10);
		if (key == 27)
			break;
	}
	return 0;
}

// Encoding watermkar to png img with key value, and change to jpeg format
cv::Mat Encoder(cv::Mat img, string watermark, int key) {

	// making sequence array for watermark
	vector<int> wm_seq(sync_seq.begin(), sync_seq.end());
	// shifting m-sequence for watermark
	for (int i = 0; i < watermark.length(); i++) {
		int message_loc = watermark[i];
		for (int j = 0; j < 256; j++) {
			if (message_loc >= message_seq.size())
				message_loc = 0;
			wm_seq.push_back(message_seq[message_loc++]);		
		}
	}

	cv::Mat block = array2block(wm_seq, 64, key);
	
	int HVS = 5;
	// watermarking (should change)
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			for (int k = 0; k < 3; k++) {
				int a = img.at<Vec3b>(i, j)[k];
				// clipping
				if (img.at<Vec3b>(i, j)[k] + block.at<char>(i % 64, j % 64) * HVS < 0)
					img.at<Vec3b>(i, j)[k] = 0;
				else if (img.at<Vec3b>(i, j)[k] + block.at<char>(i % 64, j % 64) * HVS > 255)
					img.at<Vec3b>(i, j)[k] = 255;
				else
					// watermarking
					img.at<Vec3b>(i, j)[k] += block.at<char>(i % 64, j % 64) * HVS;
				
			}
			
		}
	}
	
	return img;
}

// 1D array to 2D block matching
cv::Mat array2block(vector < int > rblock, int size, int key) {
	cv::Mat block = cv::Mat(cv::Size(size, size), CV_8S);
	//vector<int> temp(rblock.begin(), rblock.end());

	if (key == 0)
		key = 1;
	if (key == 256)
		key = 255;

	int loc = 0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			while (rblock[loc] == 0) {
				loc++;
				loc %= size;
			}

			block.at<char>(i, j) = rblock[loc];	
			match.insert({ loc, {i,j} });
			rblock[loc] = 0;
			loc += key;
			if (loc >= rblock.size())
				loc %= rblock.size();
		}
	}

	return block;
}

// Decoding image, seperating into watermark and png image from jpeg image
pair<cv::Mat, string > Decoder(cv::Mat img, int key) {
	


	return { img, "asdfasd" };
}

// 2D block to 1D array matching
vector<int> block2array(cv::Mat block, int size, int key) {
	vector<int> barray(size * size);

	pair<int, int> temp;

	for (int i = 0; i < 4096; i++) {
		temp = match[i];
		barray[i] = block.at<char>(temp.first, temp.second);
	}

	return barray;
}


