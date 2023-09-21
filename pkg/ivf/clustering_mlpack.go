package ivf

import (
	"gonum.org/v1/gonum/mat"
	"mlpack.org/v1/mlpack"
)

type mlpackClustering struct {
}

func NewMlpackClustering() Clustering {
	return &kmeansClustering{}
}

var _ Clustering = new(mlpackClustering)

func (m *mlpackClustering) ComputeClusters(clusterCnt int64, data [][]float32) (centroids [][]float32, err error) {
	param := mlpack.KmeansOptions()
	param.InitialCentroids = mat.NewDense(1, 1, nil)
	param.LabelsOnly = false
	param.MaxIterations = 1000
	param.Percentage = 0.02
	param.RefinedStart = false
	param.Samplings = 100
	param.Seed = 0
	param.Verbose = false
	param.Algorithm = "elkan"
	param.AllowEmptyClusters = true

	cols := len(data[0])
	rows := len(data) / cols

	rowCnt := int64(len(data))
	dims := int64(len(data[0]))

	vectorFlat := make([]float64, dims*rowCnt)

	for r := int64(0); r < rowCnt; r++ {
		for c := int64(0); c < dims; c++ {
			vectorFlat[(r*dims)+c] = float64(data[r][c])
		}
	}

	input := mat.NewDense(rows, cols, vectorFlat)
	centroidsFlat, _ := mlpack.Kmeans(int(clusterCnt), input, param)

	centroids = make([][]float32, clusterCnt)
	for r := 0; r < int(clusterCnt); r++ {
		center := make([]float32, cols)
		for c := 0; c < cols; c++ {
			center[c] = float32(centroidsFlat.RawMatrix().Data[r*int(dims)+c])
		}
		centroids[r] = center
	}

	return centroids, nil
}

func (m *mlpackClustering) Close() {
	//TODO implement me
	panic("implement me")
}
