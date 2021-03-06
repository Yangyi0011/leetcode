package sort

/*
插入排序：
	[]表示排好序
	第一轮： [4] 2 9 1 拿待排序的第二个数 2，插入到排好序的数列 [4]
		与排好序的数列 [4] 比较
		第一轮进行中：2 比 4 小，插入到 4 前
	第二轮： [2 4] 9 1 拿待排序的第三个数 9，插入到排好序的数列 [2 4]
		与排好序的数列 [2 4] 比较
		第二轮进行中： 9 比 4 大，不变化
	第三轮： [2 4 9] 1 拿待排序的第四个数 1，插入到排好序的数列 [2 4 9]
		与排好序的数列 [2 4 9] 比较
		第三轮进行中： 1 比 9 小，插入到 9 前
		第三轮进行中： 1 比 4 小，插入到 4 前
		第三轮进行中： 1 比 2 小，插入到 2 前
	结果： [1 2 4 9]

最好情况下，对一个已经排好序的数列进行插入排序，那么需要迭代N-1轮，并且因为每
轮第一次比较，待排序的数就比它左边的数大，那么这一轮就结束了，不需要再比较了，
也不需要交换，这样时间复杂度为：O(n)。

最坏情况下，每一轮比较，待排序的数都比左边排好序的所有数小，那么需要交换 N-1
次，第一轮需要比较和交换一次，第二轮需要比较和交换两次，第三轮要三次，第四轮
要四次，这样次数是：1 + 2 + 3 + 4 + ... + N-1，时间复杂度和冒泡排序、选择
排序一样，都是：O(n^2)。

因为是从右到左，将一个个未排序的数，插入到左边已排好序的队列中，所以插入排序，
相同的数在排序后顺序不会变化，这个排序算法是稳定的。
*/
func InsertSort(data []int) {
	n := len(data)
	if n == 0 {
		return
	}
	for i := 1; i < n; i++ {
		// 待插入的数
		deal := data[i]
		// data[0 ~ j] 是已完成排序的有序序列
		j := i - 1
		// 如果待插入的数比有序序列的最后一个数小，则需要处理
		if deal < data[j] {
			for ; j >= 0 && deal < data[j]; j-- {
				// 数据后移，给待插入的数 deal 留出空位
				data[j+1] = data[j]
			}
			// 跳出循环时待插入的数 deal >= data[j]，说明 deal 该插入到
			// data[j+1] 中
			data[j+1] = deal
		}
	}
}
