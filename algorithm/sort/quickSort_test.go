package sort

import (
	"math/rand"
	"testing"
	"time"
)

// 获取随机测试用例
func GetRandomData(n int) []int {
	// 设置随机种子
	rand.Seed(time.Now().UnixNano())
	data := make([]int, n)
	for i := 0; i < len(data); i++ {
		data[i] = rand.Intn(101)
	}
	return data
}

// 递归快排测试
func TestQuickSort(t *testing.T) {
	data := GetRandomData(20)
	t.Logf("排序前：%v\n", data)
	QuickSort(data)
	t.Logf("排序后：%v\n", data)
}

// 循环快排测试
func TestQuickSort2(t *testing.T) {
	data := GetRandomData(20)
	t.Logf("排序前：%v\n", data)
	QuickSort2(data)
	t.Logf("排序后：%v\n", data)
}

// 测试使用快速排序的思想来寻找 TopK
func TestTopKByQuickSort(t *testing.T) {
	data := GetRandomData(100)
	arr := make([]int, len(data))
	copy(arr, data)
	t.Logf("数据集：%v\n", arr)
	QuickSort(arr)
	t.Logf("排序后：%v\n", arr)
	res := TopKByQuickSort(data, 5)
	t.Logf("Top5：%v\n", res)
}