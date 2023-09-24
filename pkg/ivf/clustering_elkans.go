package ivf

import (
	"errors"
	"math"
)

type elkanClustering struct {
}

var _ Clustering = new(elkanClustering)

func NewElkanClustering() Clustering {
	return &elkanClustering{}
}
func (e *elkanClustering) ComputeClusters(clusterCnt int64, data [][]float32) (centroids [][]float32, err error) {
	if clusterCnt <= 0 || clusterCnt > int64(len(data)) {
		return nil, errors.New("Invalid cluster count")
	}

	// Randomly initialize centroids
	centroids = initializeCentroids(data, int(clusterCnt))

	maxIterations := 100 // You can adjust this as needed
	for iteration := 0; iteration < maxIterations; iteration++ {
		// Assign data points to clusters
		clusterAssignments := assignToClusters(data, centroids)

		// Calculate new centroids
		newCentroids := calculateCentroids(data, clusterAssignments, int(clusterCnt))

		// Check for convergence
		if hasConverged(centroids, newCentroids, 0.001) {
			return newCentroids, nil
		}

		centroids = newCentroids
	}

	return centroids, nil
}

func initializeCentroids(data [][]float32, clusterCnt int) [][]float32 {
	// Implement your logic to initialize centroids (e.g., random selection)
	// Here, we'll just pick the first 'clusterCnt' data points as centroids.
	centroids := make([][]float32, clusterCnt)
	copy(centroids, data[:clusterCnt])
	return centroids
}

func assignToClusters(data [][]float32, centroids [][]float32) []int {
	clusterAssignments := make([]int, len(data))

	for i, point := range data {
		minDist := float32(math.MaxFloat32)
		assignedCluster := -1

		// Initialize to an impossible value
		for j := 0; j < len(centroids); j++ {
			centroid := centroids[j]
			if len(point) != len(centroid) {
				panic("Data point and centroid must have the same dimensionality")
			}

			// Calculate the distance from the current data point to each centroid
			dist := calculateDistance(point, centroid)
			if dist < minDist {
				minDist = dist
				assignedCluster = j
			}
		}

		// Assign the data point to the nearest cluster
		clusterAssignments[i] = assignedCluster
	}

	return clusterAssignments
}

func calculateCentroids(data [][]float32, clusterAssignments []int, clusterCnt int) [][]float32 {
	// Initialize centroids and counts
	centroids := make([][]float32, clusterCnt)
	clusterCounts := make([]int, clusterCnt)

	// Calculate the sum of data points for each cluster
	for i, point := range data {
		cluster := clusterAssignments[i]
		centroid := centroids[cluster]
		if centroid == nil {
			centroid = make([]float32, len(point))
			centroids[cluster] = centroid
		}

		if len(point) != len(centroid) {
			panic("Data point and centroid must have the same dimensionality")
		}

		centroids[cluster] = addVectors(centroid, point)
		clusterCounts[cluster]++
	}

	// Calculate the new centroids as the mean of data points in each cluster
	for i := 0; i < clusterCnt; i++ {
		if clusterCounts[i] > 0 {
			centroids[i] = scaleVector(centroids[i], 1.0/float32(clusterCounts[i]))
		}
	}

	return centroids
}

func hasConverged(oldCentroids, newCentroids [][]float32, tolerance float32) bool {
	// Check for convergence by comparing old and new centroids within a tolerance
	for i := 0; i < len(oldCentroids); i++ {
		if calculateDistance(oldCentroids[i], newCentroids[i]) > tolerance {
			return false
		}
	}
	return true
}

func calculateDistance(point1, point2 []float32) float32 {
	// Calculate the Euclidean distance between two data points
	if len(point1) != len(point2) {
		panic("Data points must have the same dimensionality")
	}

	var sumSquared float32
	for i := 0; i < len(point1); i++ {
		diff := point1[i] - point2[i]
		sumSquared += diff * diff
	}

	return float32(math.Sqrt(float64(sumSquared)))
}

func addVectors(vector1, vector2 []float32) []float32 {
	// Element-wise addition of two vectors
	if len(vector1) != len(vector2) {
		panic("Vectors must have the same dimensionality")
	}

	result := make([]float32, len(vector1))
	for i := 0; i < len(vector1); i++ {
		result[i] = vector1[i] + vector2[i]
	}

	return result
}

func scaleVector(vector []float32, scaleFactor float32) []float32 {
	// Scale a vector by a given factor
	result := make([]float32, len(vector))
	for i := 0; i < len(vector); i++ {
		result[i] = vector[i] * scaleFactor
	}

	return result
}

func (e *elkanClustering) Close() {
	//TODO implement me
	panic("implement me")
}
