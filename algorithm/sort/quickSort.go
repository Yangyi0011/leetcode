package sort

import "log"

/*
快速排序：
	快速排序通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数
	据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快
	速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

步骤如下：
	1、先从数列中取出一个数作为基准数。一般取第一个数。
	2、分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它
		的左边。
	3、再对左右区间重复第二步，直到各区间只有一个数。

举一个例子：5 9 1 6 8 14 6 49 25 4 6 3。
	一般取第一个数 5 作为基准，从它左边和最后一个数使用[]进行标志，

	如果左边的数比基准数大，那么该数要往右边扔，也就是两个[]数交换，这样大于
	它的数就在右边了，然后右边[]数左移，否则左边[]数右移。
		5 [9] 1 6 8 14 6 49 25 4 6 [3]  因为 9 > 5，两个[]交换位置后，右边[]左移
		5 [3] 1 6 8 14 6 49 25 4 [6] 9  因为 3 !> 5，两个[]不需要交换，左边[]右移
		5 3 [1] 6 8 14 6 49 25 4 [6] 9  因为 1 !> 5，两个[]不需要交换，左边[]右移
		5 3 1 [6] 8 14 6 49 25 4 [6] 9  因为 6 > 5，两个[]交换位置后，右边[]左移
		5 3 1 [6] 8 14 6 49 25 [4] 6 9  因为 6 > 5，两个[]交换位置后，右边[]左移
		5 3 1 [4] 8 14 6 49 [25] 6 6 9  因为 4 !> 5，两个[]不需要交换，左边[]右移
		5 3 1 4 [8] 14 6 49 [25] 6 6 9  因为 8 > 5，两个[]交换位置后，右边[]左移
		5 3 1 4 [25] 14 6 [49] 8 6 6 9  因为 25 > 5，两个[]交换位置后，右边[]左移
		5 3 1 4 [49] 14 [6] 25 8 6 6 9  因为 49 > 5，两个[]交换位置后，右边[]左移
		5 3 1 4 [6] [14] 49 25 8 6 6 9  因为 6 > 5，两个[]交换位置后，右边[]左移
		5 3 1 4 [14] 6 49 25 8 6 6 9    两个[]已经汇总，因为 14 > 5，所以 5 和[]之前的数 4 交换位置

	第一轮切分结果：4 3 1 5 14 6 49 25 8 6 6 9
	现在第一轮快速排序已经将数列分成两个部分：
		4 3 1 和 14 6 49 25 8 6 6 9

	左边的数列都小于 5，右边的数列都大于 5。
	使用递归分别对两个数列进行快速排序。

在最好情况下，每一轮都能平均切分，这样遍历元素只要n/2次就可以把数列分成两部
分，每一轮的时间复杂度都是：O(n)。因为问题规模每次被折半，折半的数列继续递归
进行切分，也就是总的时间复杂度计算公式为：T(n) = 2*T(n/2) + O(n) = O(nlogn)。

最差的情况下，每次都不能平均地切分，每次切分都因为基准数是最大的或者最小的，
不能分成两个数列，这样时间复杂度变为了T(n) = T(n-1) + O(n) = O(n^2)，

根据熵的概念，数量越大，随机性越高，越自发无序，所以待排序数据规模非常大时，
出现最差情况的情形较少。在综合情况下，快速排序的平均时间复杂度为：O(nlogn)。
*/

// 快速排序
func QuickSort(data []int) {
	n := len(data)
	if n == 0 {
		return
	}
	log.Println("快排改进：伪尾递归优化")
	// quickSort(data, 0, n-1)
	// quickSort1(data, 0, n-1)
	quickSort2(data, 0, n-1)
}

/*
快排的分区函数
	思路：
		安照快排思想，取本轮快排区间的第一个元素 data[start] 作为基准，把区
		间内小于基准的元素都交换到基准的左边，把区间内大于基准的元素都交换到
		基准的右边，最后返回本轮分割的基准所处的下标。
*/
func partition(data []int, start, end int) int {
	// 取 data[start] 作为一轮快排的基准，所以 i 从 start+1 开始
	i, j := start+1, end
	for i < j {
		// 把大于基准的交换到右边
		if data[i] > data[start] {
			data[i], data[j] = data[j], data[i]
			j--
		} else {
			i++
		}
	}
	// 跳出循环时， i==j，基准依旧是 data[start]
	// 此时数组被分割为两个部分：data[start+1] ~ data[i-1] < data[start]，
	// data[i+1] ~ data[end] > data[start]
	// 此时还需要判断 data[i] 与 data[start] 的关系
	// 如果 data[i] >= data[start]，说明基值 data[start] 所处的下标应该在 i 之前
	if data[i] >= data[start] {
		i--
	}
	// 最后交换基值 data[start] 与 data[i]，把基准 data[start] 的值放到
	// 下标为 i 的位置去，此时一轮快排完成：data[start:i] < data[i] < data[i+1:end+1]
	data[start], data[i] = data[i], data[start]
	// i 为本轮快排的基准最后所处的位置，返回 i
	return i
}

/*
	一、普通快速排序
*/
func quickSort(data []int, start, end int) {
	if start < end {
		// 进行一轮快排，获取本轮分区基准的下标
		index := partition(data, start, end)
		// 对分区基准下标的左半部分继续进行快排
		quickSort(data, start, index-1)
		// 对分区基准下标的右半部分继续进行快排
		quickSort(data, index+1, end)
	}
}

/*
	二、快速排序改进
		1、在小规模数组的情况下，直接插入排序的效率最好，当快速排序递归部分
			进入小数组范围，可以切换成直接插入排序。
		2、使用伪尾递归减少程序栈空间占用，使得栈空间复杂度从O(logn)~log(n)
			变为：O(logn)。
*/
/*
	2.1 改进：小规模数组使用直接插入排序
*/
func quickSort1(data []int, start, end int) {
	if start < end {
		// 当数组元素个数小于 5 时使用直接插入排序
		if end-start <= 4 {
			InsertSort(data[start : end+1])
		}

		// 进行一轮快排，获取本轮分区基准的下标
		index := partition(data, start, end)
		// 对分区基准下标的左半部分继续进行快排
		quickSort1(data, start, index-1)
		// 对分区基准下标的右半部分继续进行快排
		quickSort1(data, index+1, end)
	}
}

/*
	2.2 改进：伪尾递归优化
		很多人以为这样子是尾递归。其实这样的快排写法是伪装的尾递归，不是真正
		的尾递归，因为有for循环，不是直接return QuickSort，递归还是不断地压
		栈，栈的层次仍然不断地增长。

		但是，因为先让规模小的部分排序，栈的深度大大减少，程序栈最深不会超过
		logn 层，这样堆栈最坏空间复杂度从 O(n) 降为 O(logn)。

		这种优化也是一种很好的优化，因为栈的层数减少了，对于排序十亿个整数，
		也只要：log(100 0000 0000)=29.897，占用的堆栈层数最多 30 层，比不
		进行优化，可能出现的O(n)常数层好很多。
*/
func quickSort2(data []int, start, end int) {
	for start < end {
		// 进行一轮快排，获取分区下标
		index := partition(data, start, end)

		// 那边元素少先排哪边
		if index-start < end-index {
			// 先排左边
			quickSort2(data, start, index-1)
			start = index + 1
		} else {
			// 先排右边
			quickSort2(data, index+1, end)
			end = index - 1
		}
	}
}

/*
	三、循环实现
		把递归函数改为用栈来实现。
*/
func QuickSort2(data []int) {
	log.Println("循环快排：")
	n := len(data)
	if n == 0 {
		return
	}
	stack := make([]int, 0)
	// 先入栈的是 end，后入栈的是 start
	stack = append(stack, n-1)
	stack = append(stack, 0)
	for len(stack) > 0 {
		// 先弹出的是 start，后弹出的是 end
		start := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		end := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		// 进行一轮快排，获取分区下标
		index := partition(data, start, end)

		// 右边范围入栈
		if index+1 < end {
			stack = append(stack, end)
			stack = append(stack, index+1)
		}
		// 左边范围入栈
		if start < index-1 {
			stack = append(stack, index-1)
			stack = append(stack, start)
		}
	}
}


/* 
	四、利用快排分区思想来获取 TopK
		在每一轮快排分区之后，我们都会得到 data[:i-1] < data[i] < data[i+1:] 
		的数据，其中 data[i] 是分区完成后基准的下标。
		由此，如果分区完成后 len(data)-i == k，则 data[i:] 就是我们要找的 TopK
*/
func TopKByQuickSort(data []int, k int) []int {
	n := len(data)
	if n == 0 {
		return []int{}
	}
	stack := make([]int, 0)
	// 先入栈的是 end，后入栈的是 start
	stack = append(stack, n-1)
	stack = append(stack, 0)
	for len(stack) > 0 {
		// 先弹出的是 start，后弹出的是 end
		start := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		end := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		// 进行一轮快排，获取分区下标
		index := partition(data, start, end)
		
		// 较大的区域的数据量刚好是 k 个，则它们就是 topk，直接返回
		if n - index == k {
			return data[index:]
		}
		// 如果较大的区域的数据量小于 k，则需要往左找，否则往右找
		if n - index < k {
			// 左边范围入栈
			if index-1 > start {
				stack = append(stack, index-1)
				stack = append(stack, start)
			}
		} else {
			// 右边范围入栈
			if index+1 < end {
				stack = append(stack, end)
				stack = append(stack, index+1)
			}
		}
	}
	return []int{}
}