package sort

import (
	"testing"
)

func TestInsertSort(t *testing.T) {
	data := GetRandomData(20)
	t.Logf("排序前：%v\n", data)
	InsertSort(data)
	t.Logf("排序后：%v\n", data)
}