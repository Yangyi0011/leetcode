package main

import (
	"fmt"
	"container/heap"
	"time"
	"math/rand"
)

/* 
========================== 1、堆排序 =========================
	1、堆的概念
		堆是把数组元素看作是一颗二叉树（非完全二叉树），按根节点最大、最小
		的方式可以把数组数据组织为大根堆和小根堆。
			大根堆：
				每一个根节点的值总是比它的左右子节点的值大
			小根堆
				每一个根节点的值总是比它的左右子节点的值小
			堆的根节点 i 和它的左右子节点：
				 i
				/ \
			 2*i+1 2*i+2
	2、堆排序：
		原理：
			先构建初始堆，然后按升序用大根堆，降序用小根堆的方式，把堆顶
			元素（最大/最小值）与数组的最后一个元素（n-1）进行交换，交换
			后数组的最后一个元素有序，此时堆结构可能被破坏，需要调整前 
			n-1 个元素组成的堆（最后一个元素已经有序不用管），接着再把堆
			顶元素与数组的倒数第二个元素（n-2）进行交换，继续调整堆结构，
			重复上述操作，直到第一个数组元素操作完成为止。
	3、复杂度
		时间复杂度：O(nlog2n)
			n 是数组元素个数。
		空间复杂度：O(n)
*/
// 构建大根堆
// data 堆数据
func buildMaxHeap(data []int) {
	n := len(data)
	if n == 0 {
		return
	}
	// 最后一个非叶子节点下标
	lastFatherIndex := n / 2 - 1
	// 从最后一个非叶子节点开始向上处理每一个父节点，一直处理到根节点
	for i := lastFatherIndex; i >= 0; i -- {
		// 调整堆结构
		maxHeadFix(data, i, n)
	}
}

// 调整大根堆的堆结构
// data 堆数
// fatherIndex 从哪一个父节点开始调整
// n 需要调整的堆元素个数
func maxHeadFix(data []int, fatherIndex int, n int) {
	// 记录当前父节点的值
	fatherValue := data[fatherIndex]

	// i := 2*fatherIndex+1 当前父节点的第一个子节点
	// i = 2*i+1 处理下一个父节点
	for i := 2*fatherIndex+1; i < n; i = 2*i+1 {
		// 找到当前父节点的最大子节点下标
		if i < n - 1 && data[i] < data[i + 1] {
			i ++
		}
		// 若当前父节点的值是最大的，则无需处理直接跳出循环
		if data[fatherIndex] > data[i] {
			break
		}
		// 否则把最大子节点交换到父节点的位置
		data[fatherIndex] = data[i]
		// 【关键】修改父节点下标
		// 交换过后可能会导致子节点的堆结构被破坏，需要调整子堆
		fatherIndex = i
	}
	// 将父节点的值放入调整后 fatherIndex 所处的位置
	// 这样处理其实就是子节点上浮，而父节点下沉，不使用多次交换以提升效率
	data[fatherIndex] = fatherValue
}

// 堆排序-升序
func heapSortASC(data []int) {
	// 构建堆
	buildMaxHeap(data)
	n := len(data)
	for i := n - 1; i > 0; i -- {
		// 将堆顶元素和堆底元素交换，即把当前堆的最大元素换至数组尾部
		// 从而保证数组尾部有序
		data[0], data[i] = data[i], data[0]
		// 交换后堆结构可能被破坏，需要调整堆结构
		// i 可以让已经排序好的元素不再进行堆调整处理
		maxHeadFix(data, 0, i)
	}
}

// 构建小根堆
// data 堆数据
func buildMinHeap(data []int) {
	n := len(data)
	if n == 0 {
		return
	}
	lastFatherIndex := n / 2 - 1
	for i := lastFatherIndex; i >= 0; i -- {
		minHeapFix(data, i, n)
	}
}
// 调整小根堆
// data 堆数据
// fatherIndex 从哪一个父节点开始调整
// n 需要调整的堆元素个数
func minHeapFix(data []int, fatherIndex int, n int) {
	// 从当前父节点的第一个子节点开始处理
	// i = 2*i+1 处理下一个父节点
	for i := 2 * fatherIndex + 1; i < n; i = 2*i+1 {
		// 找到子节点中的最小值
		if i < n - 1 && data[i] > data[i + 1] {
			i ++
		}
		// 如果父节点是最小值，则不用处理直接跳出循环
		if data[fatherIndex] < data[i] {
			break
		}
		// 否则把最小子节点上浮到父节点的位置
		data[fatherIndex], data[i] = data[i], data[fatherIndex]
		// 【重要】父节点下标下沉
		fatherIndex = i
	}
}

// 堆排序-逆序
func heapSortDESC(data []int) {
	// 构建小根堆
	buildMinHeap(data)
	n := len(data)
	for i := n - 1; i > 0; i -- {
		// 将堆顶元素和堆底元素交换，即把当前堆的最小元素换至数组尾部
		// 从而保证数组尾部有序
		data[0], data[i] = data[i], data[0]
		// 交换后堆结构可能被破坏，需要调整堆结构
		// i 可以让已经排序好的元素不再进行堆调整处理
		minHeapFix(data, 0, i)
	}
}

func heapSortTest() {
	data := []int{7, 3, 8, 5, 1, 2}

	// 升序排序
	// buildMaxHeap(data)
	// heapSortASC(data)
	// fmt.Println(data)

	// 逆序排序
	heapSortDESC(data)
	fmt.Println(data)
}

/* 
========================== 2、小根堆处理 topK 问题 =========================
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。
你可以按 任意顺序 返回答案。

示例 1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例 2:
输入: nums = [1], k = 1
输出: [1]

提示：
    1 <= nums.length <= 10^5
    k 的取值范围是 [1, 数组中不相同的元素的个数]
    题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的

进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/top-k-frequent-elements
*/
/* 
	方法一：小根堆【手写小根堆】
	思路：
		首先遍历整个数组，并使用哈希表记录每个数字出现的次数，并形成一个
		「出现次数数组」hash。找出原数组的前 k 个高频元素，就相当于找出
		「出现次数数组」hash 的前 k 大的值。
		使用「出现次数数组」hash 的前 k 个元素构建长度为 k 的小根堆，接
		着遍历 hash 的后 n-k 个元素，对比每一个元素 hash[i] 与小根堆的
		堆顶元素，如果 hash[i] 大于堆顶元素，则把堆顶元素替换为 hash[i]，
		接着调整小根堆，重复上述操作，直到 hash 遍历完成。
	时间复杂度：O(n*logk)
	空间复杂度：O(k)
*/
func topKFrequent(nums []int, k int) []int {
	hash := make(map[int]int, 0)
	for _, v := range nums {
		hash[v] ++
	}
	heap := make([][2]int, 0)
	// 标记堆是否初始化
	init := false
	for key, value := range hash {
		if len(heap) < k {
			heap = append(heap, [2]int{key, value})
			continue
		}
		// 初始化堆
		if !init {
			buildHeap(heap)
			init = true
		}
		if value > heap[0][1] {
			// 替换堆顶元素和调整堆结构
			heap[0] = [2]int{key, value}
			heapFix(heap, 0, k)
		}
	}
	res := make([]int, k)
	for i := 0; i < k; i ++ {
        // 升序输出
		res[k - i - 1] = heap[i][0]
	}
	return res
}

// 构建小顶堆
func buildHeap(data [][2]int) {
	n := len(data)
	if n == 0 {
		return
	}
    // 从最后一个非叶子节点开始处理，一直处理到根节点
	lastFatherIndex := n/2 - 1
	for i := lastFatherIndex; i >= 0; i -- {
		heapFix(data, i, n)
	}
}

// 调整堆结构
// data 堆数据
// fatherIndex 从哪一个父节点开始调整
// n 需要调整的堆元素个数
func heapFix(data [][2]int, fatherIndex int, n int) {
	// 2 * fatherIndex + 1 从当前父节点的第一个子节点开始处理
    // i = 2*i+1 处理下一个父节点
	for i := 2 * fatherIndex + 1; i < n; i = 2*i + 1 {
		// 找到子节点中的最小值
		if i < n - 1 && data[i][1] > data[i + 1][1] {
			i ++
		}
		// 如果父节点是最小值，则不用处理直接跳出循环
		if data[fatherIndex][1] < data[i][1] {
			break
		}
		// 否则把最小子节点上浮到父节点的位置
		data[fatherIndex], data[i] = data[i], data[fatherIndex]
		// 【重要】父节点下标下沉
		fatherIndex = i
	}
}

/* 
	方法二：小根堆【使用接口】
	时间复杂度：O(n*logn)
	空间复杂度：O(k)
*/
func topKFrequent2(nums []int, k int) []int {
	hash := make(map[int]int, 0)
	for _, v := range nums {
		hash[v] ++
	}
	// 初始化堆
	myHeap := &MyHeap{}
	heap.Init(myHeap)
	for key, value := range hash {
		// 注意，都是用 heap 包的函数来调用的
		heap.Push(myHeap, [2]int{key, value})
		// 如果堆内元素大于 k，则弹出堆顶元素
		if myHeap.Len() > k {
			heap.Pop(myHeap)
		}
	}
	res := make([]int, k)
	for i := 0; i < k; i ++ {
		res[k - i - 1] = (*myHeap)[i][0]
	}
	return res
}

// 自定义堆结构，继承 heap 包的 Interface 接口
type MyHeap [][2]int
// 继承自 sort.Interface
func (h MyHeap) Len() int {
	return len(h)
}
func (h MyHeap) Less(i, j int) bool {
	return h[i][1] < h[j][1]
}
func (h MyHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
// 继承 heap.Interface
func (h *MyHeap) Push(x interface{}) {
	*h = append(*h, x.([2]int))
}
func (h *MyHeap) Pop() interface{} {
	x := (*h)[h.Len() - 1]
	*h = (*h)[: h.Len() - 1]
	return x
}

/* 
	方法三：基于快速排序
	思路：
		根据快排的性质：每一趟快排都将大于基值 K 的数放在 K 的右边，小于
		K 的数放在 K 的左边，由此每次快速排序中的划分过程定能找到一个全
		部大于左边元素的一个值 nums[i]。
		
		如果 len(nums)-1-i = K，那么这个值和它右边的所有元素就是 topK；
		如果 len(nums)-1-i > K，那么对右边的元素继续划分排序；
		如果 len(nums)-1-i < K，那么对左边的元素继续划分排序。
	时间复杂度：O(n^2)
		其中 n 是数组长度。
		设处理长度为 n 的数组的时间复杂度为 f(n)。由于处理的过程包括一次
		遍历和一次子分支的递归，最好情况下，有 f(N)=O(N)+f(N/2)，根据
		主定理，能够得到 f(N)=O(N)。
		最坏情况下，每次取的中枢数组的元素都位于数组的两端，时间复杂度退
		化为 O(N^2)。但由于我们在每次递归的开始会先随机选取中枢元素，故
		出现最坏情况的概率很低。
		平均情况下，时间复杂度为 O(n)。
	空间复杂度：O(n)
*/
func topKFrequent3(nums []int, k int) []int {
	hash := map[int]int{}
    for _, num := range nums {
        hash[num]++
    }
    data := [][2]int{}
    for key, value := range hash {
        data = append(data, [2]int{key, value})
    }
	// 完成交换后，topK 就是 data[index:]
	index := quickSort(data, 0, len(data) - 1, k)
	res := make([]int, 0)
	for i := index; i < len(data); i ++ {
		res = append(res, data[i][0])
	}
    return res
}
// 通过快速排序原理来寻找 topK
// 返回第 k 大的数所在的下标
// data 数据
// start、end 每一轮快排的起始和终止下标
// k topK
func quickSort(data [][2]int, start, end, k int) int {
	// 选择随机基值的下标
	rand.Seed(time.Now().UnixNano())
    picked := rand.Int() % (end - start + 1) + start;
	data[picked], data[start] = data[start], data[picked]
	
	// 记录基值
	pivot := data[start][1]
	left, right := start, end
	// 一轮快排
	for (left < right) {
		// 先从右往左找小于基值的数，跳过大于等于基值的数
		for (left < right && data[right][1] >= pivot) {
			right --
		}
		// 找到了小于基值的数，把它交换到基值的前面
		if (data[right][1] < pivot) {
			data[left], data[right] = data[right], data[left]
			// 越过刚刚交换过来的元素，从左边开始找
			left ++
		}
		// 从左往右找大于基值的数，跳过小于等于基值的数
		for (left < right && data[left][1] <= pivot) {
			left ++
		}
		// 找到了大于基值的数，把它交换到基值的后面
		if (data[left][1] > pivot) {
			data[left], data[right] = data[right], data[left]
			// 越过刚刚交换过来的元素，从右边开始找
			right --
		}
	}
	// 获取基值下标往后的元素个数 rank
	rank := len(data) - left
	// 如果 rank 刚好是 k 个，说明找到了 topK 开始的下标
	if rank == k {
		return left
	}
	// rank < k，需要往左找
	if rank < k {
		return quickSort(data, start, left - 1, k)
	}
	// rank > k 往右找
	return quickSort(data, right + 1, end, k)
}

func topKFrequentTest() {
	nums := []int{1,1,1,2,2,3}
	k := 2
	res := topKFrequent3(nums, k)
	fmt.Println(res)
}

func main() {
	// heapSortTest()
	topKFrequentTest()
}