#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
  cv::Mat image;
  image = cv::imread("test.png", 1);

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);
  return 0;
}
