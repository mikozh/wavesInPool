#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

float waveFunction(float r, float t, int maxImageSize);

/* Subroutine that creates wave map layer - no lookup table*/
void makeWaveMap(Mat& image);

/* Subroutine that blends wave map layer and original image - no lookup table*/
void blendWaveAndImage(Mat& sourceImage, Mat& targetImage, Mat& waveMap);

/* Subroutine that creates wave map layer - lookup table*/
void makeWaveMapLUT(Mat& image);

/* Subroutine that blends wave map layer and original image - lookup table*/
void blendWaveAndImageLUT(Mat& sourceImage, Mat& targetImage, Mat& waveMap);


int main(int argc, char* argv[])
{
   Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);        // Original image
   Mat frameDst = frame.clone();                            // Distorted image
   double t1, t2;
   double timeWave = 0.E0;
   double timeBlen = 0.E0;


   namedWindow("Wave Map",CV_WINDOW_NORMAL);
   namedWindow("Distorted Image",CV_WINDOW_NORMAL);
   namedWindow("Original Image",CV_WINDOW_NORMAL);

   Mat waveMap(frame.rows, frame.cols, CV_8U, Scalar::all(0));

   int frameCount = 1;

   while (1) {
      int timingPeriod = 20;

      int pressedKey = waitKey(10);

      if ( pressedKey == 27 ) {
         cout << "esc key is pressed by user" << endl;
         break; 
      }

      t1 = (double)getTickCount();

//      makeWaveMap(waveMap);
      makeWaveMapLUT(waveMap);

      t2 = (double)getTickCount();

      timeWave+= (t2 - t1)/getTickFrequency();

//      blendWaveAndImage(frame, frameDst, waveMap);
      blendWaveAndImageLUT(frame, frameDst, waveMap);

      t1 = (double)getTickCount();

      timeBlen+= (t1 - t2)/getTickFrequency();

      imshow("Wave Map", waveMap);
      imshow("Distorted Image", frameDst);
      imshow("Original Image", frame);

      frameCount++;

      if (frameCount % timingPeriod == 0) {
         cout << "Average Wave Generation Time: " << timeWave*0.05E0 << endl;
         cout << "Average Layers Blending Time: " << timeBlen*0.05E0 << endl;
         cout << endl;
         frameCount = 1;
         timeWave = 0.E0;
         timeBlen = 0.E0;
      }
   }
   return 0;
}

void blendWaveAndImage(Mat& sourceImage, Mat& targetImage, Mat& waveMap)
{
   static float rFactor = 1.33; // refraction factor of water

   for (int i = 1; i < sourceImage.rows-1; i++) {
      for (int j = 1; j < sourceImage.cols-1; j++) {
         float alpha, beta;

         float xDiff = waveMap.at<uchar>(i+1, j) - waveMap.at<uchar>(i, j);
         float yDiff = waveMap.at<uchar>(i, j+1) - waveMap.at<uchar>(i, j);

         alpha = atan(xDiff);
         beta = asin(sin(alpha)/rFactor);
         int xDisplace = cvRound(tan(alpha - beta)*waveMap.at<uchar>(i, j));

         alpha = atan(yDiff);
         beta = asin(sin(alpha)/rFactor);
         int yDisplace = cvRound(tan(alpha - beta)*waveMap.at<uchar>(i, j));

         Vec3b Intensity = sourceImage.at<Vec3b>(i,j);

         /* Check whether displacement fits the image size */
         int dispNi = i + xDisplace;
         int dispNj = j + yDisplace;
         dispNi = (dispNi > sourceImage.rows || dispNi < 0 ? i : dispNi);
         dispNj = (dispNj > sourceImage.cols || dispNj < 0 ? j : dispNj);

         Intensity = sourceImage.at<Vec3b>(dispNi, dispNj);

         targetImage.at<Vec3b>(i,j) = Intensity;
      }
   }
}

void blendWaveAndImageLUT(Mat& sourceImage, Mat& targetImage, Mat& waveMap)
{
   static float rFactor = 1.33; // refraction factor of water
   static float dispLUT[512];      //Lookup table for displacement
   static int nDispPoint = 512;

   for (int i = 0; i < nDispPoint; i++) {
      float diff = saturate_cast<float>(i - 255);
      float alpha = atan(diff);
      float beta = asin(sin(alpha)/rFactor);
      dispLUT[i] =  tan(alpha - beta);
   }
   nDispPoint = 0;

   for (int i = 1; i < sourceImage.rows-1; i++) {
      for (int j = 1; j < sourceImage.cols-1; j++) {
         int xDiff = waveMap.at<uchar>(i+1, j) - waveMap.at<uchar>(i, j);
         int yDiff = waveMap.at<uchar>(i, j+1) - waveMap.at<uchar>(i, j);

         int xDisplace = cvRound(dispLUT[xDiff+255]*waveMap.at<uchar>(i, j));

         int yDisplace = cvRound(dispLUT[yDiff+255]*waveMap.at<uchar>(i, j));

         Vec3b Intensity = sourceImage.at<Vec3b>(i,j);

         /* Check whether displacement fits the image size */
         int dispNi = i + xDisplace;
         int dispNj = j + yDisplace;
         dispNi = (dispNi > sourceImage.rows || dispNi < 0 ? i : dispNi);
         dispNj = (dispNj > sourceImage.cols || dispNj < 0 ? j : dispNj);

         Intensity = sourceImage.at<Vec3b>(dispNi, dispNj);

         targetImage.at<Vec3b>(i,j) = Intensity;
      }
   }
}

float waveFunction(float r, float t, int maxImageSize)
{
   static float L = maxImageSize/8.0;          // Wave length
   const float twoPI = 2.0*3.1415;
   const float c = 0.5;                         // Damping coefficient in time domain

   return (exp(-t*c)*cos(t*twoPI)*cos(r*twoPI/L));
}


void makeWaveMap(Mat& image)
{
   float simulPeriod = 10.0;     // Period of simulation
   static float time = 0.0;
   const float dt = 0.05;        // Time step
   float poolDepth = 20.0;
   int maxImageSize = image.cols > image.rows ? image.cols : image.rows;

   for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
         float radius = sqrt((i - image.rows/2)*(i - image.rows/2) + \
                             (j - image.cols/2)*(j - image.cols/2));
         float z = (1.0 + waveFunction(radius, time, maxImageSize))*poolDepth;
         image.at<uchar>(i, j) = saturate_cast<uchar>(z);
      }
   }

   time+= dt;
   time*= (time < simulPeriod);
}

void makeWaveMapLUT(Mat& image)
{
   float simulPeriod = 10.0;     // Period of simulation
   static float time = 0.0;
   const float dt = 0.05;        // Time step
   float poolDepth = 20.0;
   int nLUT = image.cols > image.rows ? image.cols : image.rows;
   int maxImageSize = nLUT;
   float waveFuncLUT[nLUT];

   for (int i = 0; i < nLUT; i++) {
      float radius = saturate_cast<float>(i);
      waveFuncLUT[i] = waveFunction(radius, time, maxImageSize);
   }

   for (int i = 0; i < image.rows; i++) {
      for (int j = 0; j < image.cols; j++) {
         float radius = sqrt((i - image.rows/2)*(i - image.rows/2) + \
                             (j - image.cols/2)*(j - image.cols/2));
         int iRad = cvRound(radius);
         float dR = radius - saturate_cast<float>(iRad);
         float wF = waveFuncLUT[iRad] + (waveFuncLUT[iRad+1] - waveFuncLUT[iRad])*dR;
         float z = (1.0 + wF)*poolDepth;
         image.at<uchar>(i, j) = saturate_cast<uchar>(z);
      }
   }

   time+= dt;
   time*= (time < simulPeriod);
}
