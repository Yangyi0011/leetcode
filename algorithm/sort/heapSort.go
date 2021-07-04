package sort

/*
堆排序：
	堆排序是利用堆这种数据结构而设计的一种排序算法，堆排序是一种选择排序，它
	的最坏、最好、平均时间复杂度均为O(nlogn)，它也是不稳定排序。

	首先来看下堆的结构：
		大顶堆：
					 50
				   /    \
				 45      40
			   /   \    /  \
			 20    25  35  30
		    /  \
		  10    15

		小顶堆：
		  			    10
					  /    \
					20      15
				  /    \   /  \
				25     50 30   40
			   /  \
			  35   45

	对应于数组：
		大顶堆： 50 45 40 20 25 35 30 10 15
		  下标： 0  1  2  3  4  5  6  7  8
		小顶堆： 10 20 15 25 50 30 40 35 45

	由此可见，该数组从逻辑上讲就是一个堆结构，我们用简单的公式来描述一下堆的
	定义就是：
		大顶堆：arr[i] >= arr[2i+1] && arr[i] >= arr[2i+2]
		小顶堆：arr[i] <= arr[2i+1] && arr[i] <= arr[2i+2]
	其中，arr[i] 是堆的非叶子节点，而 arr[2i+1]、arr[2i+2] 是 arr[i] 的左右
	子节点。

堆排序的基本思想及步骤：
	一、构造初始堆。
		将给定的无序列表构造成一个大顶堆（一般升序采用大顶堆，降序采用小顶堆）。
			1、我们从最后一个非叶子节点开始，从下到上把数组调整成大顶堆结构。
				最后一个非叶子节点下标为：arr[len(arr)/2-1]
			2、判断当前非叶子节点 arr[i] 与它两个子节点中较大的那个节点的大小
				关系，如果 arr[i] 是最大的，则不用继续调整，否则交换 arr[i] 
				和较大的子节点，然后 i 改为较大子节点的下标，向下继续调整堆。
	二、将堆顶元素与末尾元素进行交换，使末尾元素最大。
	三、继续调整堆，使其满足堆的定义。接着再将堆顶元素与末尾元素交换，得到第二
		大元素。如此反复进行交换、重建、交换。
*/

// 堆排序
func HeapSort(arr []int) {
	if len(arr) == 0 {
		return
	}
	// 构建大顶堆
	buildMaxHeap(arr, len(arr))
	// 把最大的堆顶元素交换到堆的末尾，然后继续调整堆
	// i 表示当前堆的末尾下标
	for i := len(arr) - 1; i >= 0; i -- {
		arr[0], arr[i] = arr[i], arr[0]
		fixMaxHeap(arr, 0, i)
	}
}

// 从下到上构建大顶堆
// n：堆元素个数
func buildMaxHeap(arr []int, n int) {
	// 从最后一个非叶子节点开始向上浮动构建大顶堆
	for i := n/2 - 1; i >= 0; i-- {
		fixMaxHeap(arr, i, n)
	}
}

// fixMaxHeap 从上到下调整堆数组，使其满足大顶堆的定义
// i：从哪一个非叶子节点的下标开始调整
// length：需要调整到哪一个下标结束
func fixMaxHeap(arr []int, i int, length int) {
	// K 从当前非叶子节点的左子节点开始，向下一个非叶子节点的左子节点不断处理
	for k := 2*i+1; k < length; k = 2*k+1 {
		// 如果左子节点小于右子节点，则 k 指向右子节点
		// 即 k 指向较大的子节点
		if k+1 < length && arr[k] < arr[k+1] {
			k++
		}
		// 如果父节点比它的子节点都大，则不用调整
		if arr[i] > arr[k] {
			break
		}
		// 否则把较大的子节点交换到父节点的位置
		arr[i], arr[k] = arr[k], arr[i]
		// 向下继续处理下一个非叶子节点
		i = k
	}
}

// 从下到上构建小顶堆
func buildMinHeap(arr []int, n int) {
	// 从最后一个非叶子节点开始向上浮动构建小顶堆
	for i := n/2-1; i >= 0; i -- {
		fixMinHeap(arr, i, n)
	}
}
// fixMinHeap 从上到下调整堆数组，使其满足小顶堆的定义
// i：从哪一个非叶子节点的下标开始调整
// length：需要调整到哪一个下标结束
func fixMinHeap(arr []int, i, length int) {
	for k := 2*i+1; k < length; k = 2*k+1 {
		// k 指向当前非叶子节点的较小子节点
		if k + 1 < length && arr[k] > arr[k+1] {
			k ++
		}
		// 如果当前非叶子节点是最小的，则不用处理
		if arr[i] < arr[k] {
			break
		}
		// 否则交换当前非叶子节点和它较小的子节点
		arr[i], arr[k] = arr[k], arr[i]
		// 继续向下处理下一个非叶子节点
		i = k
	}
}

// 利用小顶堆从数据列表中获取 topK
func TopK(arr []int, k int) []int {
	// 先用前 k 个元素构建小顶堆
	buildMinHeap(arr, k)
	// 接着从第 k+1 个元素开始，对比 arr[i] 和 堆顶元素 arr[0] 的大小关系，
	// 如果 arr[i] > arr[0]，则 arr[0] = arr[i]，继续调整堆结构
	for i := k+1; i < len(arr); i ++ {
		if arr[i] > arr[0] {
			arr[0] = arr[i]
			fixMinHeap(arr, 0, k)
		}
	}
	// 返回前 k 个元素
	return arr[:k]
}