# opencv_assignment2_barcode
barcode recognition for camera and image

/*********************************************************************************************
 * compile with:
 * g++ -O3 -o barcode barcode.cpp -std=c++11 `pkg-config --cflags --libs opencv`
*********************************************************************************************/


/*
 *                                SPECIFICATION
 *  The programe aims to decode color barcode image. You can specify a static image
 *  for decoding, or capture image via camera without specifing any parameter.
 *  The processing workflow is
 *  #1 Find the circle of the image
 *  #2 Calculate the angle that the image need to be rotated to allign the circle
 *  #3 Allign the circle, so that the lines in the circle are either vertical or parallel
 *     to the horizon. If the circle is too small, scale the circle to be larger and move 
 *     it to the center of the image.
 *  #4 Find the lines in the circle, we need 48*48 lines to generate 47*47 sample points.
 *     If the lines found are not enough, try to interpolate new lines.
 *  #5 The circle has been devided by 48 * 48 lines, find the 47 * 47 sample points
 *  #6 Find the top direction by checking the colors of neighbours of the center point
 *  #7 decode sample points following the order which is calculated based on top direction
 */

Any question, contact raoyonghui0630@gmail.com
