// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

float displacementOptimized(int dx, int dy)
{
	float squaredDisplacement = dx * dx + dy * dy;
	float displacement = sqrt(squaredDisplacement);
	return displacement;
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{						
	int mod = (featureWidth * 2 + 1) % 4;
	int heightDiff = imageHeight - featureHeight;
	int widthDiff = imageWidth - featureWidth;

	for (int z = 0; z < imageHeight*imageWidth; z++){
		depth[z] = 0;
	}

	#pragma omp parallel for
	for (int y = 0; y < imageHeight; y ++)
	{

	float savedMinDisplacement = 0.0;

		for (int x = 0; x < imageWidth; x ++)
		{	
			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
			if ((y < featureHeight) || (y >= heightDiff) || (x < featureWidth) || (x >= widthDiff))
			{
				depth[y * imageWidth + x] = 0;
				continue;
			}

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				/* Skip feature boxes that dont fit in the displacement box. */
				if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight) 
				{
					continue;
				}

				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					/* Skip feature boxes that dont fit in the displacement box. */
					if (x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;

					__m128 temp = _mm_setzero_ps();

					__m128 squaredDifferenceVector = _mm_setzero_ps();
					float temp2[4];

					int leftY = y + -featureHeight;
					int rightY = y + dy + -featureHeight;

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY ++)
					{
						
						int leftYImage = leftY * imageWidth;
						int rightYImage = rightY * imageWidth;

						for (int boxX = -featureWidth; boxX <= (featureWidth - mod); boxX += 4)
						{
							int leftX = x + boxX;
							int rightX = x + dx + boxX;
							
							//is it quicker to do the subs manually and load once into muL?

							temp = _mm_sub_ps(_mm_loadu_ps(&left[leftYImage + leftX]), _mm_loadu_ps(&right[rightYImage + rightX]));
							temp = _mm_mul_ps(temp, temp);
							squaredDifferenceVector = _mm_add_ps(temp, squaredDifferenceVector);

						}
						if (mod > 0) {

							int leftCalc = leftYImage + x + featureWidth;
							int rightCalc = rightYImage + x + dx + featureWidth;
							float difference = 0;
							switch(mod) {
							    case 1 :
									difference = left[leftCalc] - right[rightCalc];
									squaredDifference += difference * difference;
							        break;
							    case 3 :
								    difference = left[leftCalc] - right[rightCalc];
									squaredDifference += difference * difference;
									difference = left[leftCalc - 1] - right[rightCalc - 1];
									squaredDifference += difference * difference;
									difference = left[leftCalc - 2] - right[rightCalc - 2];
									squaredDifference += difference * difference;
							}
						}

						leftY++;
						rightY++;
					}
					
					_mm_storeu_ps(temp2, squaredDifferenceVector); 	//stores 128-bit vector a at pointer p
					 
					squaredDifference += temp2[0];
					squaredDifference += temp2[1];
					squaredDifference += temp2[2];
					squaredDifference += temp2[3];
					

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || (minimumSquaredDifference > squaredDifference) || ((minimumSquaredDifference == squaredDifference) && (displacementOptimized(dx, dy) < savedMinDisplacement)))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
						savedMinDisplacement = displacementOptimized(minimumDx, minimumDy);
					}
				}
			}
			

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (maximumDisplacement != 0)
			{

				depth[y * imageWidth + x] = savedMinDisplacement;
			}
		}
	}
}
