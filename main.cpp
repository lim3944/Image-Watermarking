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

// weiner filter
#include "wiener.h"

using namespace std;
using namespace cv;

// png image to jpg with watermark & key
cv::Mat Encoder(cv::Mat, string, int);
cv::Mat array2block(vector<int>, int, int);

// jpg image to png seperating image and watermark with key
string Decoder(cv::Mat, int);
vector<int> block2array(cv::Mat, int);

cv::Mat pixelVar(cv::Mat);

cv::Mat g,hvsMat;

string mes_temp = "1 0 0 1 0 1 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 1 0 1 1 0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1 1 0 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 0 1 1 1 0 0 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 0 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 1 0 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 1 0 1 1 1 1";
string sync_temp = "0 1 0 0 0 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 0 0 0 1 1 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 1 0 0 1 0 0 0 0 1 1 1 1 1 0 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 1 0 1 1 0 0 1 0 1 1 1 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 0 1 1 0 0 0 0 1 1 0 1 0 1 1 0 1 1 1 0 1 0 0 0 1 0 1 0 1 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 1 1 0 1 1 0 1 0 0 1 1 1 1 0 0 1 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 1 1 0 1 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 1 0 1 1 0 1 1 0 0 0 1 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 1 0 0 1 1 1 1 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 1 1 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1 1 0 0 0 1 0 0 0 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 0 1 0 1 0 1 0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 1 0 1 0 0 1 1 0 1 0 1 1 1 1 1 0 0 1 1 0 0 0 1 1 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 0 0 1 0 0 1 1 1 0 1 1 0 0 1 0 1 0 1 1 1 0 1 1 1 1 0 1 0 1 0 0 0 1 1 1 1 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1 0 1 1 1 1 0 1 0 0 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0 1 0 1 0 1 0 0 1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 1 1 0 1 0 0 1 1 1 0 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 0 0 0 1 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 0 1 1 0 0 1 1 1 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 1 0 1 1 0 1 1 1 0 0 0 0 1 1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 1 0 1 1 0 1 0 0 1 0 1 1 1 0 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 1 1 0 0 0 1 1 0 1 1 1 0 1 1 0 0 0 0 1 1 1 1 0 0 1 0 0 1 1 1 0 0 1 0 1 1 0 0 1";

// m-sequence for sync, message
vector<int> sync_seq;
vector<int> message_seq;

map<int, pair<int, int>> match;

int main() {

	string watermark;
	int key = 0;
	cv::Mat img;
	img = cv::imread("6.png"); // image name

	// Watermark and key value setting
	while (watermark.length() != 12) {
		std::printf("Enter 12 letters watermark :");
		cin >> watermark;
	}

	std::printf("Enter key value : ");
	cin >> key;
	
	// m-sequenc for sync and message
	for (int i = 0; i < sync_temp.length(); i++) {
		if (sync_temp[i] == ' ')
			continue;
		sync_seq.push_back(sync_temp[i] == '1' ? -1 : 1);
	}

	for (int i = 0; i < mes_temp.length(); i++) {
		if (mes_temp[i] == ' ')
			continue;
		message_seq.push_back(mes_temp[i] == '1' ? -1 : 1);
	}

	//cv::imshow("original", img);
	cv::Mat input;
	img.copyTo(input);
	img.copyTo(g);

	// Encoding
	cv::Mat img_encoded;
	img_encoded = Encoder(input, watermark, key);

	//cv::imshow("watermarked img", img_encoded);

	cv::imwrite("6.jpg", img_encoded);

	// Decoding
	string wm_decoded;
	wm_decoded = Decoder(img_encoded, key);
	
	cout << "decoded message : " << wm_decoded << endl;

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

	// array to block
	cv::Mat block = array2block(wm_seq, 64, key);
	
	hvsMat = pixelVar(img);
	
	int HVS = 0;
	
	// watermarking process
	for (int i = 0; i < img.rows; i++) {
		std::printf("watermakring...%d%%\n",i*100/1024);
		for (int j = 0; j < img.cols; j++) {
			for (int k = 0; k < 3; k++) {

				// set HVS value as alpha in watermark
				int a = img.at<Vec3b>(i, j)[k];
				HVS = hvsMat.at<Vec3b>(i, j)[k] / 10;
				if (HVS <= 0)
					HVS = 1;
				if (HVS > 10)
					HVS = 10;
				hvsMat.at<Vec3b>(i, j)[k] = HVS;

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
string Decoder(cv::Mat img, int key) {
	
	// denoising with wiener filter
	cv::Mat denoised;

	cv::Mat rgb_origin[3];
	split(img, rgb_origin);
	cv::Mat rgb_decoded[3];

	// denosing image using wiener-filter
	WienerFilter(rgb_origin[0], rgb_decoded[0], cv::Size(3, 3));
	WienerFilter(rgb_origin[1], rgb_decoded[1], cv::Size(3, 3));
	WienerFilter(rgb_origin[2], rgb_decoded[2], cv::Size(3, 3));

	string message;
	message.resize(12);

	int mes_value[12] = { 0, };
	int sum_max = 0;

	cv::Mat subimg(cv::Size(1024, 1024), CV_8S);

	// Extracting watermark message
	for (int channel = 0; channel < 3; channel++) {
		for (int block_x = 0; block_x < 1024; block_x+= 64) {
			printf("channel %d processing...%d%%\n", channel, block_x * 100 / (1024));
			for (int block_y = 0; block_y < 1024; block_y += 64) {
				// extract message per block
				for (int i = block_x; i < block_x + 64; i++) {
					// noised image - denoised image = noise
					for (int j = block_y; j < block_y +64; j++) {
						int rgb_sub = rgb_origin[channel].at<unsigned char>(i, j) - rgb_decoded[channel].at<unsigned char>(i, j);
						
						if (rgb_sub > 0)
							rgb_sub = 1;
						else if (rgb_sub < 0)
							rgb_sub = -1;
						// clipping detection
						else if (rgb_sub == 0) {
							if (rgb_decoded[channel].at<unsigned char>(i, j) == 255) {
								rgb_sub = 1;
							}
							else if (rgb_decoded[channel].at<unsigned char>(i, j) == 0) {
								rgb_sub = -1;
							}
						}

						subimg.at<char>(i, j) = rgb_sub;
					}
				}

				vector<int> barray(64 * 64);
				pair<int, int> temp;

				// array, matrix matching
				for (int k = 0; k < 4096; k++) {
					temp = match[k];
					barray[k] = subimg.at<char>(temp.first+block_x, temp.second+block_y);
				}

				// sync check
				int sync_sum = 0;
				for (int k = 0; k < 1024; k++) {
					sync_sum += barray[k] * sync_seq[k];
				}
				if (sync_sum < 600)
					continue;

				// message decoding
				for (int k = 0; k < 12; k++) {
					for (int m = 0; m < 128; m++) {
						int mes_sum = 0;
						for (int index = 0; index < 256; index++)
							mes_sum += message_seq[(index+m)%256] * barray[1024 + k*256+index];
						
						// save most accurate(highest) value as watermark
						if (mes_sum > 150 && mes_sum > mes_value[k]) {
							message[k] = m;
							mes_value[k] = mes_sum;
							break;
						}
					}
				}

			}
		}
	}


	return message;
}

// 2D block to 1D array matching
vector<int> block2array(cv::Mat block, int key) {
	vector<int> barray(64 * 64);

	pair<int, int> temp;

	for (int i = 0; i < 4096; i++) {
		temp = match[i];
		barray[i] = block.at<char>(temp.first, temp.second);
	}

	return barray;
}

// calcuate variacne of each pixel in 5X5 block
cv::Mat pixelVar(cv::Mat img) {
	cv::Mat rgb[3];
	split(img, rgb);
	int max = 0;

	for (int i = 0; i < 1024; i++) {
		if (i % 62 == 0)
			std::printf("calculating...%d %%\n",i*100/1024);
		for (int j = 0; j < 1024; j++) {
			int sum[3] = { 0, };
			int mean[3] = { 0, };
			int var[3] = { 0, };
			int cnt = 0;
			
			for (int dx = -2; dx <= 2; dx++) {
				for (int dy = -2; dy <= 2; dy++) {
					if (i + dx >= 0 && j + dy >= 0 && i + dx < 1024 && j + dy < 1024) {
						sum[0] += img.at<Vec3b>(i + dx, j + dy)[0];
						sum[1] += img.at<Vec3b>(i + dx, j + dy)[1];
						sum[2] += img.at<Vec3b>(i + dx, j + dy)[2];
						cnt++;
					}
				}
			}

			mean[0] = sum[0] / cnt;
			mean[1] = sum[1] / cnt;
			mean[2] = sum[2] / cnt;

			for (int dx = -2; dx <= 2; dx++) {
				for (int dy = -2; dy <= 2; dy++) {
					if (i + dx >= 0 && j + dy >= 0 && i + dx < 1024 && j + dy < 1024) {
						var[0] += (img.at<Vec3b>(i + dx, j + dy)[0] - mean[0])/cnt * (img.at<Vec3b>(i + dx, j + dy)[0] - mean[0]);
						var[1] += (img.at<Vec3b>(i + dx, j + dy)[1] - mean[1])/cnt  * (img.at<Vec3b>(i + dx, j + dy)[1] - mean[1]);
						var[2] += (img.at<Vec3b>(i + dx, j + dy)[2] - mean[2])/cnt * (img.at<Vec3b>(i + dx, j + dy)[2] - mean[2]);
					}
				}
			}

			rgb[0].at<char>(i, j) = (int)sqrt((float)var[0]);
			rgb[1].at<char>(i, j) = (int)sqrt((float)var[1]);
			rgb[2].at<char>(i, j) = (int)sqrt((float)var[2]);

			if (max < var[0])
				max = var[0];
			if (max < var[1])
				max = var[1];
			if (max < var[2])
				max = var[2];

		}
	}

	std::printf("max = %d\n", max);

	cv::Mat out;
	cv::merge(rgb,3, out);
	return out;
}
