// CS 61C Fall 2015 Project 4

// Checklist:
// Factor out common calculations and keep the result.
// Avoid recalculating values you have already calculated.

// Avoid expensive operations like divisons or square roots, when unnecessary.
// Unroll loops so that you can use SSE intrinsics and perform 4 operations at a time.
// Don't be afraid to restructure the loops or change the ordering of some operations.
// Padding matrices may give better performance for odd sized matrices.
// Use OpenMP to multithread your code.
// Avoid forking and joining too many times, and avoid critical sections, barriers, and single sections.
// Avoid false sharing and data dependencies between processors.


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

float displacementOptimized(int dx, int dy)
{
	float squaredDisplacement = dx * dx + dy * dy;
	float displacement = sqrt(squaredDisplacement);
	return displacement;	
}


void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{

// int savedDy = 0;
// int savedDx = 0;
// float savedDisplacement = 0.0;

// int savedMinDy = 0;
// int savedMinDx = 0;
float savedMinDisplacement = 0.0;

int heightDiff = imageHeight - featureHeight;
int widthDiff = imageWidth - featureWidth;

	/* The two outer for loops iterate through each pixel */
	for (int y = 0; y < imageHeight; y++)
	{
		for (int x = 0; x < imageWidth; x++)
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
			// MOVED: 'y + dy -/+ featureHeight' check outside of dx loop as they are dependent on outer loop not inner - should cut down on # of checks. 
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

					
					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					int leftY = y + -featureHeight;
					int rightY = y + dy + -featureHeight;
					
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						int leftX = x + -featureWidth;
						int rightX = x + dx + -featureWidth;
						
						for (int boxX = -featureWidth; boxX <= featureWidth; boxX++)
						{
							float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
							squaredDifference += difference * difference;

							leftX++;
							rightX++;
						}
						
						leftY++;
						rightY++;						
					}

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					
					ADDED: Only recalculates displacement when mins are altered.
					CHANGED: Order of IF check so dx dy displacement is only calculated if first two checks are false 
					STILL TO DO: Switch statement instead of if check? Will this offer any speed up? Cuts out ALU... 
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

			ADDED: Checks to see if mindx/mindy have changed, if not then no need to recalculate displacement.
			STILL TO DO: Figure out whether dx and dy can change independently; if not then no need to check them seperately. 
			*/
			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[y * imageWidth + x] = 0;
				}
				else
				{
					// if (minimumDx != savedMinDx) 
					// {
					// 	savedMinDisplacement = displacementOptimized(minimumDx, minimumDy);
					// 	savedMinDx = minimumDx;
					// 	savedMinDy = minimumDy;
					// }
					
					// else if (minimumDy != savedMinDy) 
					// {
					// 	savedMinDisplacement = displacementOptimized(minimumDx, minimumDy);
					// 	savedMinDy = minimumDy;
					// }

					depth[y * imageWidth + x] = savedMinDisplacement;
				}
			}
			else
			{
				depth[y * imageWidth + x] = 0;
			}
		}
	}

}
