package sort

import (
	"testing"
)

func TestHeapSort(t *testing.T) {
	data := GetRandomData(20)
	t.Logf("排序前：%v\n", data)
	HeapSort(data)
	t.Logf("排序后：%v\n", data)
}

func TestTopK(t *testing.T) {
	data := GetRandomData(100)
	arr := make([]int, len(data))
	copy(arr, data)
	t.Logf("数据集：%v\n", arr)
	HeapSort(arr)
	t.Logf("排序后：%v\n", arr)
	res := TopK(data, 5)
	t.Logf("Top5：%v\n", res)
}