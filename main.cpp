#include <opencv2/opencv.hpp>
#include <iostream>
#include "Algorithm.h"

using namespace cv;
using namespace std;

int matsize(vector<Mat>& vm)
{
	return vm.size();
}
RNG rng( 12345);
int main()
{
//	ConnComponetLabelDemo();
//	CCLabelingDemo();
//	SelectRegionDemo();
//	FMTmatchDemo();
//	FastHazeRemovalDemo();
	CylinderExpansionTest();
//	StatFeatureInfoDemo();
//	SelectShapeDemo();
//	floodfilltest();
}