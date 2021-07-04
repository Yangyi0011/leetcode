package main

/* 
============== 剑指 Offer 11. 旋转数组的最小数字 ==============
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个
递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 
为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

示例 1：
输入：[3,4,5,1,2]
输出：1

示例 2：
输入：[2,2,2,0,1]
输出：0

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof
*/
/* 
	方法一：二分查找【模板三】
	思路：
		虽然旋转数组整体无序，但是我们在中间点对其二分之后，nums[0:mid] 和
		nums[mid:n] 中总有一部分是有序的，如此我们就可以在有序部分使用二分
		查找。
		在每一次二分查找过程中，我们选取当前区间的最后一个元素 nums[R] 来和 
		nums[mid] 做对比：
			如果 nums[R] > nums[mid]，说明 nums[mid] 的右边有序，但最小
			值在 nums[mid] 的左边，我们需要往左找（R = mid），否则往右找
			（L = mid）。

		注意：输入数据会有重复元素，我们需要过滤掉重复元素的影响。
	时间复杂度：O(logn)
		平均时间复杂度为 O(log⁡n)，其中 n 是数组 numbers 的长度。如果数组是
		随机生成的，那么数组中包含相同元素的概率很低，在二分查找的过程中，大
		部分情况都会忽略一半的区间。而在最坏情况下，如果数组中的元素完全相同，
		那么过滤重复元素的循环就需要执行 n 次，时间复杂度为 O(n)。
	空间复杂度：O(1)
*/
func minArray(numbers []int) int {
	n := len(numbers)
	if n == 0 {
		return -1
	}
	L, R := 0, n - 1
	for L + 1 < R {
		// 过滤掉重复元素
		for L < R && numbers[R] == numbers[R - 1] {
			R --
		}
		for L < R && numbers[L] == numbers[L + 1] {
			L ++
		}
		mid := L + ((R - L) >> 1)
		// 说明右边有序，但最小值在左边，否则最小值在右边
		if numbers[R] > numbers[mid] {
			R = mid
		} else {
			L = mid
		}
	}
	if numbers[L] < numbers[R] {
		return numbers[L]
	}
	return numbers[R]
}