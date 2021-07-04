package array

/* 
	数组
*/

/* 
========================== 1、寻找数组的中心索引 ==========================
给你一个整数数组 nums，请编写一个能够返回数组 “中心索引” 的方法。
数组 中心索引 是数组的一个索引，其左侧所有元素相加的和等于右侧所有元素相加的和。
如果数组不存在中心索引，返回 -1 。如果数组有多个中心索引，应该返回最靠近左边的那一个。

注意：中心索引可能出现在数组的两端。

示例 1：
输入：nums = [1, 7, 3, 6, 5, 6]
输出：3
解释：
中心索引是 3 。
左侧数之和 (1 + 7 + 3 = 11)，
右侧数之和 (5 + 6 = 11) ，二者相等。

示例 2：
输入：nums = [1, 2, 3]
输出：-1
解释：
数组中不存在满足此条件的中心索引。

示例 3：
输入：nums = [2, 1, -1]
输出：0
解释：
中心索引是 0 。
索引 0 左侧不存在元素，视作和为 0 ；
右侧数之和为 1 + (-1) = 0 ，二者相等。

提示：
    nums 的长度范围为 [0, 10000]。
    任何一个 nums[i] 将会是一个范围在 [-1000, 1000]的整数。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/yf47s/
*/
/* 
	方法一：迭代
	思路：
		先计算数组元素的总和 sum，然后从左往右遍历数组并计算当前和 left，
		如果 sum - left - nums[i] == left，说明 i 即为数组中心。
	时间复杂度：O(n)
		我们需要遍历数组两次。
	空间复杂度：O(1)
*/
func pivotIndex(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	sum := 0
	for i := 0; i < n; i ++ {
		sum += nums[i]
	}
	leftSum := 0
	for i := 0; i < n; i ++ {
		if leftSum == sum - leftSum - nums[i] {
			return i
		}
		leftSum += nums[i]
	}
	return -1
}

/* 
========================== 2、搜索插入位置 ==========================
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

示例 1:
输入: [1,3,5,6], 5
输出: 2

示例 2:
输入: [1,3,5,6], 2
输出: 1

示例 3:
输入: [1,3,5,6], 7
输出: 4

示例 4:
输入: [1,3,5,6], 0
输出: 0

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cxqdh/
*/
/* 
	方法一：二分查找模板三
		思路：
		在搜索空间内做比较判断：
			小于等于搜索空间内最左边的返回 L，处于搜索空间内部的返回 R，
			大于搜索空间最右边的返回 R + 1
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func searchInsert(nums []int, target int) int {
	n := len(nums)
	if n == 0 || target < nums[0] {
		return 0
	}
	L, R := 0, n - 1
	mid := 0
	for L + 1 < R {
		mid = L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid
		} else {
			R = mid
		}
	}
	// 退出循环后，搜索空间剩下两个元素
	if target <= nums[L] {
		// 小于等于搜索空间较小元素的
		return L
	} else if target <= nums[R]{
		// 大于搜索空间较小元素，且小于等于搜索空间较大元素的
		return R
	} else {
		// 大于搜索空间较大元素的
		return R + 1
	}
}

/* 
========================== 3、合并区间 ==========================
以数组 intervals 表示若干个区间的集合，其中单个区间为 
intervals[i] = [starti, endi] 。请你合并所有重叠的区间，
并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。

示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

示例 2：
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c5tv3/
*/
type intss [][]int
func(this intss) Len() int {
	return len(this)
}
func(this intss) Less(i, j int) bool {
	return this[i][0] < this[j][0]
}
func (this intss) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
/* 
	方法一：排序处理
	思路：
		对二维数组的区间左端进行预排序处理，如此左端相近的区间就都被按升序进行
		了排序，在遍历处理时，我们就只需要对比当前区间的左端和上一区间的右端
		来判断是否需要进行区间合并，不需要合并的直接添加到结果集，如需区间合并，
		则合并区间的左端等于上一区间的左端，合并区间的右端等于 max(preR，R)
		preP：上一区间的右端，R：当前区间的右端
	时间复杂度：O(nlogn)
		n 表示区间的个数，我们需要先对区间进行排序，耗时O(logn)，之后再遍历
		所有区间进行合并，耗时O(n)，故总的时间复杂度为 O(nlogn)
	空间复杂度：O(logn)
		快速排序需要 O(logn) 的空间复杂度
*/
func merge(intervals [][]int) [][]int {
	n := len(intervals)
	if n == 0 {
		return [][]int{}
	}
	// 按区间左端进行排序
	sort.Sort(intss(intervals))
	res := make([][]int, 0)
	for i := 0; i < n; i ++ {
		// 获取当前区间的左右两端
		L, R := intervals[i][0], intervals[i][1]

		// 如果结果集为空 或 当前区间的左端 大于 上一区间的右端，直接添加
		if len(res) == 0 || L > res[len(res) - 1][1] {
			res = append(res, intervals[i])
		} else {
			// 当前的右端与上一区间的右端合并，取较大值
			res[len(res) - 1][1] = max(res[len(res) - 1][1], R)
		}
	}
	return res
}

/* 
========================== 4、寻找旋转排序数组中的最小值 ==========================
假设按照升序排序的数组在预先未知的某个点上进行了旋转。例如，数组 
[0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] 。
请找出其中最小的元素。

示例 1：
输入：nums = [3,4,5,1,2]
输出：1

示例 2：
输入：nums = [4,5,6,7,0,1,2]
输出：0

示例 3：
输入：nums = [1]
输出：1

提示：
    1 <= nums.length <= 5000
    -5000 <= nums[i] <= 5000
    nums 中的所有整数都是 唯一 的
    nums 原来是一个升序排序的数组，但在预先未知的某个点上进行了旋转

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c3ki5/
*/
/* 
	方法一：二分查找
	思路：
		已知数组已经预先按升序排序，只是可能被旋转了一下。如果是正常的排序数组，
		我们是很容易就对其使用二分查找的，但是现在数组可能被旋转了，该怎么办呢？
			观察：
				 L    mid    R
				 ↓     ↓     ↓
				[0,1,2,4,5,6,7] -> 正常升序
				[4,5,6,7,0,1,2] -> 有旋转
				[7,0,1,2,4,5,6] -> 有旋转
				[1,2,4,5,6,7,0] -> 有旋转
			对于正常升序的数组，数组的最后一个元素总是大于中间元素，所以
				最小值应该往左找。
			对于旋转数组，无论怎么旋转，如果我们选取最后一个元素作为每一轮二分查找的key，
			则都只有以下两种情况：
				1、最后一个元素大于中间元素，此时最小值在左边，往左找
				2、最后一个元素小于中间元素，此时最小值在右边，往右找
			只要一直循环就可以找到最小值，由此我们依旧可以旋转数组使用二分查找。
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func findMin(nums []int) int {
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L + 1 < R {
		mid := L + ((R - L) >> 1)
		if nums[R] > nums[mid] {
			R = mid
		} else {
			L = mid
		}
	}
	// 返回最小值
	if nums[L] < nums[R] {
		return nums[L]
	}
	return nums[R]
}

