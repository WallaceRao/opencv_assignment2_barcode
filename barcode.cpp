#include <stdio.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <set>
#include <map>
#include <stdlib.h>


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


using namespace std ;
using namespace cv ;
using namespace chrono;


#define MpixelB(image,x,y) ((uchar *) (((image).data) + (y) * ((image).step)))[(x)*(image.channels())]
#define MpixelG(image,x,y) ((uchar *) (((image).data) + (y) * ((image).step)))[(x)*(image.channels()) + 1]
#define MpixelR(image,x,y) ((uchar *) (((image).data) + (y) * ((image).step)))[(x)*(image.channels()) + 2]

#define sampleSizeX 47
#define sampleSizeY 47


Mat frame; // Image from camera
Point circleCenter;
double rotateAngle = 0.0;
float radius = 0.0;

vector<Vec4i> lines;
vector<float> sampleY;
vector<float> sampleX;

int sampleValue[sampleSizeX][sampleSizeY];
char outValue[825] = {0};

int topLeftIndexX = 0;
int topLeftIndexY = 0;

int sampleSizeByRow[47] = {4, 8, 10, 12, 13, 15, 16, 17, 18, 18, 19 ,20, 20, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23,
                        0, 23, 23, 23, 23, 22, 22, 22, 22, 21, 21, 20, 20, 19, 18, 18, 17, 16, 15, 13, 12, 10, 8, 4};

char encodingarray[64]={' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s',
                        't','u','v','x','y','w','z','A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','X','Y','W','Z','0','1','2','3','4','5','6',
                        '7','8','9','.'};


bool findCenter(Mat src);




// calibrate the color and return the integer of the closest color sample
int colorToInt(int R, int G, int B)
{
    int ret = 0;
    ret =  (R-128) > 0? 4 : 0;
    ret += (G-128) > 0? 2 : 0;
    ret += (B-128) > 0? 1 : 0;
    return ret;
}


// Move the image
void translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
}


/*
 * Rotate Image to allign the circle, if it is too small, move the circle to the center of
 * the image if needed, then enlarge the circle.
 */

bool myAffine(Mat src, Mat &outImg)
{
    // Rotate
    Mat rot_mat( 2, 3, CV_32FC1 );
    rot_mat = getRotationMatrix2D( circleCenter, rotateAngle, 1 );
    warpAffine( src, outImg, rot_mat, outImg.size() );

    
    // Move, and enlarge the circle
    float enlargeRatio = (min(outImg.cols,outImg.rows) / (radius * 2)) * 0.8;
    if(enlargeRatio > 1)
    {
        
        int imageCenterX = outImg.cols / 2;
        int imageCenterY = outImg.rows / 2;
        int yOffset = imageCenterY - circleCenter.y;
        int xOffset = imageCenterX - circleCenter.x;
        // Move the circle to the center of the image
        translateImg(outImg, xOffset, yOffset);
        radius = enlargeRatio * radius;
        resize(outImg, outImg, cv::Size(outImg.cols * enlargeRatio,outImg.rows * enlargeRatio), 0, 0, CV_INTER_LINEAR);
        //pick up the circle
        int xBegin = max(0, (outImg.cols - src.cols)/2);
        int yBegin = max(0, (outImg.rows - src.rows)/2);
        outImg = cv::Mat(outImg, cv::Rect(xBegin, yBegin, src.cols, src.rows)).clone();
        // Update the center point after affine transformation.
        findCenter(outImg);
    }
  
    return true;
}


// Find the center of the circle
bool findCenter(Mat src)
{
    Mat src_gray;
    if( !src.data )
        return -1;

    /// Convert it to gray
    cvtColor( src, src_gray, CV_BGR2GRAY );
    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
    vector<Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 100, 200, 100, src_gray.rows/10, src_gray.rows/2 );
    if(!circles.size())
        return false;

    // Select the first largest circle
    Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
    radius = cvRound(circles[0][2]) * 1.05;

    circleCenter.x = cvRound(circles[0][0]);
    circleCenter.y = cvRound(circles[0][1]);
    return true;
}



//Get the angle of a line.
double getAngle(Point p1, Point p2)
{
    double angle = -1;
    float disX = p1.x - p2.x;
    float disY = p1.y - p2.y;

    float tanV = disY/disX;
    float radian = atan(tanV);

    double pi = CV_PI;
    angle = radian * 180.0 / pi;
    while (angle < 0)
        angle += 90.0; // Make sure the angle is between [0, 90)

    return angle;
}



// Calculate the distance between the point and the circlecenter

float distanceToCenter(Point point)
{
    return sqrt((point.x - circleCenter.x) * (point.x - circleCenter.x) +
                (point.y - circleCenter.y) * (point.y - circleCenter.y));
}



// Calculate the angle of the circle for rotating

bool calculateAngle(Mat src)
{
    Mat dst;
    Canny(src, dst, 50, 200, 3);
    lines.clear();
    HoughLinesP(dst, lines, 1, CV_PI/180, 150, 50, 10 );
    vector<float> angles;
    int angleCount = 0;
    double totalAngle = 0;
    double preAngle = -1;

    float base_angle = -1;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        if(distanceToCenter(Point(l[0], l[1])) > radius ||
           distanceToCenter(Point(l[2], l[3])) > radius )
        {
            continue;  // The line is not within the circle, drop it.
        }
        double angle = getAngle(Point(l[0], l[1]), Point(l[2], l[3]));
        angles.push_back(angle);
    }
    angleCount = angles.size();
    if(!angleCount)
        return false;
    sort(angles.begin(), angles.end());
    float midAngle = angles[angleCount/2];
    /*
     * Sometines we get  nearly vertical angles. for example, one is 90 degree
     * and next one is 1 degree, they can not be calculated together since the
     * average of them is 45 degree, we do not want to rotate the image by 45
     * degree, instead we want to rotate by (90 + 1 +90)/2 = 90.5 degree or
     * (90 + 1 - 90) = 0.5 degree, so when the next angle is more than 45 degrees
     * larger or smaller than the first angle, minus 90 degrees or add 90 degrees.
     */
     for(int i=0; i < angles.size(); i ++)
    {
        if(angles[i]- midAngle > 45)
        {
            angles[i] -= 90;
        } else if (angles[i]- midAngle < -45)
        {
            angles[i] += 90;
        }
    }
    /*
     * Filter the angles, calculate the middle value of all angles,
     * For each angle, if it is more than 10 degrees smaller or larger,
     * it is treated as a noisy angle, otherwise it is a valid angle.
     * Use all valid angles to calculate an average angle
     */
    sort(angles.begin(), angles.end());  // Sort again and
    midAngle = angles[angleCount/2];
    totalAngle = 0;
    angleCount = 0;
    for(int i=0; i < angles.size(); i ++)
    {
        if(abs(angles[i]- midAngle) < 10)
        {
            totalAngle += angles[i];
            angleCount++;
        }
    }
    if(angleCount)
        rotateAngle = totalAngle/angleCount;
    else
        return false;
    return true;
}



// Find all lines in the circle.

bool findLines(Mat src, Mat &outImg)
{
    Mat dst, cdst2;
    outImg = Mat::zeros( src.rows, src.cols, src.type() );

    Canny(src, dst, 50, 200, 3);
    lines.clear();
    vector<Vec4i> tempLines;

    HoughLinesP(dst, tempLines, 1, CV_PI/180, 90, 50, 10 );
    for( size_t i = 0; i < tempLines.size(); i++ )
    {
        Vec4i l = tempLines[i];
        if(distanceToCenter(Point(l[0], l[1])) > radius ||
           distanceToCenter(Point(l[2], l[3])) > radius )
        {
            continue;  // The line is not within the circle, drop it.
        }
        double angle = getAngle(Point(l[0], l[1]), Point(l[2], l[3]));
        if(abs(abs(angle - 45) - 45.0) > 1.0) // The angle should be within (-1, 1) or (89, 91)
            continue;
        lines.push_back(tempLines[i]);
        line(outImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
    }
    return true;
}



/*
 * If no enough lines are found by "HoughLinesP" for generating sample points,
 * interpolate new lines according to the distance of adjacent lines.
 * combinedHorizonLines and combinedVerticalLines have been sorted.
 */

void interpolateLines(vector<float> &combinedVerticalLines,
                      vector<float> &combinedHorizonLines)
{
    float standardDistance = radius * 2.0 / (sampleSizeX + 4);

    int vSize = combinedVerticalLines.size();
    int hSize = combinedHorizonLines.size();
    // We can insert 5 lines at most for each direction.
    if(vSize < (sampleSizeX - 35) || hSize < (sampleSizeY - 35))
    {
        return;
    }
    if(vSize < sampleSizeX + 1)
    {
        for(int i = 1; i < vSize; i ++)
        {
            float position = combinedVerticalLines[i -1];
            float distance = combinedVerticalLines[i] - combinedVerticalLines[i -1];
            while(distance > 1.7 * standardDistance)
            {
                // These two lines are too far from each other, insert a new line
                position += standardDistance;
                combinedVerticalLines.push_back(position);
                distance -= standardDistance;
            }
        }
    }
    if(hSize < sampleSizeY + 1)
    {
        for(int i = 1; i < hSize; i ++)
        {
            float position = combinedHorizonLines[i -1];
            float distance = combinedHorizonLines[i] - combinedHorizonLines[i -1];
            while(distance > 1.7 * standardDistance)
            {
                // These two lines are too far from each other, insert a new line
                position += standardDistance;
                combinedHorizonLines.push_back(position);
                distance -= standardDistance;
            }
        }
    }
    sort(combinedHorizonLines.begin(), combinedHorizonLines.end());
    sort(combinedVerticalLines.begin(), combinedVerticalLines.end());

    vSize = combinedVerticalLines.size();
    hSize = combinedHorizonLines.size();
    // If lines are still not enough, try to insert lines at the head or the end.
    float head = combinedVerticalLines[0], tail = combinedVerticalLines[vSize -1];
    int centerX = circleCenter.x;
    int centerY = circleCenter.y;
    while(vSize < sampleSizeX + 1)
    {
        if(abs(head - circleCenter.x) < abs(tail - circleCenter.x)) // Insert a line before the head
        {
            head -= standardDistance;
            combinedVerticalLines.push_back((int)head);
        } else {                                            // Insert a line after the tail
            tail += standardDistance;
            combinedVerticalLines.push_back((int)tail);
        }
        vSize++;
    }
    head = combinedHorizonLines[0];
    tail = combinedHorizonLines[hSize -1];
    while(hSize < sampleSizeY + 1)
    {
        if(abs(head - circleCenter.y) < abs(tail - circleCenter.y)) // Insert a line before the head
        {
            head -= standardDistance;
            combinedHorizonLines.push_back((int)head);
        } else {                                            // Insert a line after the tail
            tail += standardDistance;
            combinedHorizonLines.push_back((int)tail);
        }
        hSize++;
    }
    sort(combinedHorizonLines.begin(), combinedHorizonLines.end());
    sort(combinedVerticalLines.begin(), combinedVerticalLines.end());
}



/*
 *  Calculate the sample points, the circle has been devided by the vertical lines
 *  and horizon lines, the middle x values of each two adjacent vertical lines
 *  are the possible x values of sample points. The average y values of each two
 *  adjacent horizon lines are the possible y values of sample points.
 *  The all possible (x,y)s are the position of sample points.
 *  Note that some lines may be very near with each other, they represent
 *  the same line and should be combined to one line.
 *  For example, three lines are x=0.0, x= 0.1, x = 1, then the first two lines
 *  should be combined to x = 0.5.
 */

bool calculateSamplePoint()
{
    vector<float> horizonLines;
    vector<float> verticalLines;
    vector<float> combinedHorizonLines;
    vector<float> combinedVerticalLines;
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        float disX = l[0] - l[2];
        float disY = l[1] - l[3];
        if(abs(disX) > abs(disY) * 10)   // The line is paralell with y axis
            horizonLines.push_back((l[1] + l[3]) / 2.0);
        else if(abs(disY) > abs(disX) * 10)  // The line is paralell with x axis
            verticalLines.push_back((l[0] + l[2]) / 2.0);
    }
    sort(verticalLines.begin(), verticalLines.end());
    sort(horizonLines.begin(), horizonLines.end());

    float startPoint = 0;
    int i = 0, m = 0;
    while (i < verticalLines.size())
    {
        float x = verticalLines[i];
        float totalX = 0.0;
        startPoint = x;
        int j = 0;
        /*
         * There may be a couple of points which are near with the start point,
         * they represent the same line, so calculate an average value of them.
         */
        while(i+j < verticalLines.size() && verticalLines[i+j] - startPoint < radius /50 )
        {
            totalX += verticalLines[i+j];
            j ++;
        }
        combinedVerticalLines.push_back(totalX/j);
        i = i + j;
    }
    i = 0;
    startPoint = 0;
    while (i < horizonLines.size())
    {
        float y = horizonLines[i];
        float totalY = 0.0;
        startPoint = y;
        int j = 0;
        while(i+j < horizonLines.size() && horizonLines[i+j] - startPoint < radius/50 )
        {
            totalY += horizonLines[i+j];
            j ++;
        }
        combinedHorizonLines.push_back(totalY/j);
        i = i + j;
    }

    /*
     * To generate 47 * 47 sample points, we should have 48 * 48 lines.
     * Insert new lines if current lines are not enough.
     */
    if(combinedVerticalLines.size() < sampleSizeX + 1 ||
       combinedHorizonLines.size() < sampleSizeY + 1)
        interpolateLines(combinedVerticalLines, combinedHorizonLines);
    if(combinedVerticalLines.size() != sampleSizeX + 1 ||
       combinedHorizonLines.size() != sampleSizeY + 1)
    {
        printf("Can not gernerate sample points since the lines are not correct, %d, %d\n",
                combinedVerticalLines.size(),  combinedHorizonLines.size());
        return false;
    }
    for(int i = 0; i < combinedVerticalLines.size() - 1; i ++)
    {
        sampleX.push_back((combinedVerticalLines[i] + combinedVerticalLines[i + 1]) / 2);
    }
    for(int i = 0; i < combinedHorizonLines.size() - 1; i ++)
    {
        sampleY.push_back((combinedHorizonLines[i] + combinedHorizonLines[i + 1]) / 2);
    }
    if(sampleX.size() != sampleSizeX ||  sampleY.size() != sampleSizeY)
    {
        return false;
    }
    return true;

}


//  Highlight sample point
void drawSamplePoint(Mat src, Mat &outImg)
{
    outImg = src.clone();
    for(int i = 0; i < sampleX.size(); i ++)
        for(int j = 0; j < sampleY.size(); j ++)
        {
            float x = sampleX[i];
            float y = sampleY[j];
            circle( outImg, Point(x,y), 2, Scalar(0,255,0), -1, 8, 0 );
        }
}



// Calculate the top direction of the circle
int calculateTopDirection(Mat src)
{
    struct Index
    {
        int x;
        int y;
    };

    bool topFound = false;
    Index indexs[4];

    indexs[0].x = sampleX[sampleSizeX/2];     // the upper sample point of the center
    indexs[0].y = sampleY[sampleSizeY/2 - 1];

    indexs[1].x = sampleX[sampleSizeX/2 - 1];     // the left sample point of the center
    indexs[1].y = sampleY[sampleSizeY/2];

    indexs[2].x = sampleX[sampleSizeX/2 ];     // the below sample point of the center
    indexs[2].y = sampleY[sampleSizeY/2 + 1];

    indexs[3].x = sampleX[sampleSizeX/2 + 1];    // the right sample point of the center
    indexs[3].y = sampleY[sampleSizeY/2];
    int i = 0;

    for (i = 0; i < 4; i ++)   // if point indexs[i] and pint indexs[i+1] are both red, indexs[i] points to the top direction
    {
        int B = MpixelB(src, indexs[i].x, indexs[i].y);
        int G = MpixelG(src, indexs[i].x, indexs[i].y);
        int R = MpixelR(src, indexs[i].x, indexs[i].y);
        if(colorToInt(R, G, B) == 4)
        {
            int j = (i + 1) % 4;
            int B = MpixelB(src, indexs[j].x, indexs[j].y);
            int G = MpixelG(src, indexs[j].x, indexs[j].y);
            int R = MpixelR(src, indexs[j].x, indexs[j].y);
            if(colorToInt(R, G, B) == 4)
            {
                topFound =true;
                break;
            }
        }
    }
    if(!topFound)
    {
        return -1;
    }
    return i%4;
}


// Convert the int to a char.
char intToChar(int val)
{
    if(val < 64 && val >= 0)
        return encodingarray[val];
    printf("intToChar error, no char for integer:%d\n", val);
    return '!';
}



// Parse the sample points to a char array.
void parseSamples(Mat src, int topDirection)
{
    int valueIndex = 0;
    if(topDirection == 0)  // The top left sample point is (sampleX[0], sampleY[0])
    {
        for(int i = 0; i < 47; i ++)
        {
            int sampleSizeInThisRow = sampleSizeByRow[i];
            int start = sampleSizeX/2 - sampleSizeInThisRow;
            for(int j = start; j < sampleSizeX/2 + sampleSizeInThisRow; )
            {
                if(j  == sampleSizeX/2)  // The middle line/row has no data, so jump to next sample
                    j++;
                int B1 = MpixelB(src, (int)sampleX[j], (int)sampleY[i]);
                int G1 = MpixelG(src, (int)sampleX[j], (int)sampleY[i]);
                int R1 = MpixelR(src, (int)sampleX[j], (int)sampleY[i]);

                if(j+1 == sampleSizeX/2)  // The middle line/row has no data, so jump to next sample
                    j++;

                int B2 = MpixelB(src, (int)sampleX[j+1], (int)sampleY[i]);
                int G2 = MpixelG(src, (int)sampleX[j+1], (int)sampleY[i]);
                int R2 = MpixelR(src, (int)sampleX[j+1], (int)sampleY[i]);

                int value1 = colorToInt(R1, G1, B1);
                int value2 = colorToInt(R2, G2, B2);

                int value = value1 * 8 + value2;
                outValue[valueIndex] = intToChar(value);
                valueIndex ++;
                j += 2;
            }
        }
    }
    else if (topDirection == 1) // The top left sample point is (sampleX[0], sampleY[46])
    {
        for(int i = 0; i < 47; i ++)
        {
            int sampleSizeInThisRow = sampleSizeByRow[i];
            int start = sampleSizeY/2 + sampleSizeInThisRow;
            for(int j = start; j > sampleSizeY/2 - sampleSizeInThisRow; )
            {
                if(j  == sampleSizeY/2)  // The middle line/row has no data, move up to next sample
                    j--;
                int B1 = MpixelB(src, (int)sampleX[i], (int)sampleY[j]);
                int G1 = MpixelG(src, (int)sampleX[i], (int)sampleY[j]);
                int R1 = MpixelR(src, (int)sampleX[i], (int)sampleY[j]);

                if(j-1 == sampleSizeY/2)  // The middle line/row has no data, move up to next sample
                    j--;

                int B2 = MpixelB(src, (int)sampleX[i], (int)sampleY[j-1]);
                int G2 = MpixelG(src, (int)sampleX[i], (int)sampleY[j-1]);
                int R2 = MpixelR(src, (int)sampleX[i], (int)sampleY[j-1]);

                int value1 = colorToInt(R1, G1, B1);
                int value2 = colorToInt(R2, G2, B2);

                int value = value1 * 8 + value2;
                outValue[valueIndex] = intToChar(value);
                valueIndex ++;
                j -= 2;
            }
        }
    }
    else if (topDirection == 2) // The top left sample point is (sampleX[46], sampleY[46])
    {
        for(int i = 46; i >= 0; i --)    //Y  Movesfrom bottom to top
        {
            int sampleSizeInThisRow = sampleSizeByRow[i];
            int start = sampleSizeX/2 + sampleSizeInThisRow;
            for(int j = start; j > sampleSizeX/2 - sampleSizeInThisRow; )   // X Moves from right to left
            {
                if(j  == sampleSizeX/2)  // The middle line/row has no data, so jump to next sample
                    j--;
                int B1 = MpixelB(src, (int)sampleX[j], (int)sampleY[i]);
                int G1 = MpixelG(src, (int)sampleX[j], (int)sampleY[i]);
                int R1 = MpixelR(src, (int)sampleX[j], (int)sampleY[i]);

                if(j-1 == sampleSizeX/2)  // The middle line/row has no data, so jump to next sample
                    j--;

                int B2 = MpixelB(src, (int)sampleX[j-1], (int)sampleY[i]);
                int G2 = MpixelG(src, (int)sampleX[j-1], (int)sampleY[i]);
                int R2 = MpixelR(src, (int)sampleX[j-1], (int)sampleY[i]);

                int value1 = colorToInt(R1, G1, B1);
                int value2 = colorToInt(R2, G2, B2);

                int value = value1 * 8 + value2;
                outValue[valueIndex] = intToChar(value);
                valueIndex ++;
                j -= 2;
            }
        }
    }
    else if (topDirection == 3) // The top left sample point is (sampleX[46], sampleY[0])
    {
         for(int i = 46; i >=  0; i --)
        {
            int sampleSizeInThisRow = sampleSizeByRow[i];
            int start = sampleSizeY/2 - sampleSizeInThisRow;
            for(int j = start; j < sampleSizeY/2 + sampleSizeInThisRow; )
            {
                if(j  == sampleSizeY/2)  // The middle line/row has no data, move down to next sample
                    j++;
                int B1 = MpixelB(src, (int)sampleX[i], (int)sampleY[j]);
                int G1 = MpixelG(src, (int)sampleX[i], (int)sampleY[j]);
                int R1 = MpixelR(src, (int)sampleX[i], (int)sampleY[j]);

                if(j+1 == sampleSizeY/2)  // The middle line/row has no data, move down to next sample
                    j++;

                int B2 = MpixelB(src, (int)sampleX[i], (int)sampleY[j+1]);
                int G2 = MpixelG(src, (int)sampleX[i], (int)sampleY[j+1]);
                int R2 = MpixelR(src, (int)sampleX[i], (int)sampleY[j+1]);

                int value1 = colorToInt(R1, G1, B1);
                int value2 = colorToInt(R2, G2, B2);

                int value = value1 * 8 + value2;
                outValue[valueIndex] = intToChar(value);
                valueIndex ++;
                j += 2;
            }
        }
    }
    printf("%d chars are parsed\n", valueIndex);
}


// Init globle variables
void initVals()
{
    Point circleCenter = Point(0,0);
    rotateAngle = 0.0;
    radius = 0.0;

    lines.clear();
    sampleY.clear();
    sampleX.clear();
    topLeftIndexX = 0;
    topLeftIndexY = 0;
}


void showCam()
{
   VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
        {
            cout << "Failed to open camera" << endl;
            return;
        }
    cout << "Opened camera" << endl;
    namedWindow("1. WebCam", 1);
    namedWindow("2. Sample Points", 1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1500);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);

    cap >> frame;
    printf("frame size %d %d \n",frame.rows, frame.cols);
    int key=0;
    double fps=0.0;
    int topDirection = -1;
    bool error = false;
    char *errorMessage;
    while (true)
    {
        system_clock::time_point start = system_clock::now();
        cap >> frame;
        if( frame.empty() )
            break;
        initVals();
        error = false;
        Mat bufferImage[4];
        bufferImage[0]=frame;
        if(!findCenter(frame))
        {
            error = true;
            errorMessage = "Error: Unable to find the a circle";
            goto nextLoop;
        }
        // Mark the center for adjusting camera  by user
        circle( frame, circleCenter, 5, Scalar(255,128,0), -1, 8, 0 );

        if(!calculateAngle(bufferImage[0]))
        {
            error = true;
            errorMessage = "Error: unable to calculate the angle";
            goto nextLoop;
        }
        myAffine(bufferImage[0], bufferImage[1]);
        if(!findLines(bufferImage[1], bufferImage[2]))
        {
            error = true;
            errorMessage = "Error: unable to find correct lines in the circle";
            goto nextLoop;
        }
        if(!calculateSamplePoint())
        {
            error = true;
            errorMessage = "Error: unable to generate sample points";
            goto nextLoop;
        }

        drawSamplePoint(bufferImage[1], bufferImage[3]);
        topDirection = calculateTopDirection(bufferImage[1]);
        if(topDirection == -1)
        {
            error = true;
            errorMessage = "Error: unable to find the top direction of the bar code";
            goto nextLoop;
        }
        parseSamples(bufferImage[1],topDirection);
        outValue[824] = 0;
        printf("top direction is %d\n output result: %s\n", topDirection, outValue);

    nextLoop:
        if(error)
            putText(frame,errorMessage,cvPoint(10, 30),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(128,0,128));
        imshow("1. WebCam",  frame);
        if(bufferImage[3].rows > 0 && bufferImage[3].cols > 0)
        {
            putText(bufferImage[3],"Please refer to terminal output",cvPoint(10, 30),CV_FONT_HERSHEY_DUPLEX,1.0f,CV_RGB(128,0,128));
            imshow("2. Sample Points",  bufferImage[3]);
        }

        key=waitKey(50);
        if(key==113 || key==27)
            break;
    }
}

int main(int argc, char** argv )
{
    if ( argc != 2)
    {
        showCam();
        return 0;
    }

    Mat bufferImage[4];
    bufferImage[0]=imread(argv[1]);

    if(!findCenter(bufferImage[0]))
    {
        printf("error: can not find the center\n");
        return 0;
    }
    if(!calculateAngle(bufferImage[0]))
    {
        printf("error: can not calculate the angle\n");
        return 0;
    }
    myAffine(bufferImage[0], bufferImage[1]);
    if(!findLines(bufferImage[1], bufferImage[2]))
    {
        printf("error: can not find correct lines in the circle\n");
        return 0;
    }
    
    if(!calculateSamplePoint())
    {
        printf("error: can not calculate sample points\n");
        return 0;
    }
    drawSamplePoint(bufferImage[1], bufferImage[3]);
    int topDirection = calculateTopDirection(bufferImage[1]);
    if(topDirection == -1)
    {
        printf("error: can not find top direction of the circle\n");
        return 0;
    }
    parseSamples(bufferImage[1],topDirection);
    printf("output result:");
    for(int i = 0; i < 824; i ++)
        printf("%c", outValue[i]);
    printf("\n");
     
    imshow("1.Original Image",  bufferImage[0]);
    imshow("2.Allign and Scale Image",  bufferImage[1]);
    imshow("3.All Lines",  bufferImage[2]);
    imshow("4.Sample Point",  bufferImage[3]);
    waitKey();
    return 0 ;
}

