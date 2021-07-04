package main

import "fmt"

/*
============== 剑指 Offer 03. 数组中重复的数字 ==============
找出数组中重复的数字。
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些
数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找
出数组中任意一个重复的数字。

示例 1：
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3

限制：
2 <= n <= 100000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof
*/
/*
	方法一：哈希表法
	思路：
		遍历数组，使用一个 map[int]bool 来记录每一个数是否出现过，遇到已经
		出现过的数直接返回。
	时间复杂度：O(n)
		n 是数组元素个数，最坏情况下我们需要遍历完整个数组。
	空间复杂度：O(n)
		n 是数组元素个数，最坏情况下我们需要记录所有数字是否已经出现过。
*/
func findRepeatNumber(nums []int) int {
	hash := make(map[int]bool, 0)
	for _, v := range nums {
		if hash[v] {
			return v
		}
		hash[v] = true
	}
	return -1
}

/*
	方法二：数组重排序
	思路：
		已知长度为 n 的数组 nums 里面的所有数字都在 0~n-1 的范围内，因此当
		我们对数组元素进行排序后，如果数组中没有重复元素，则排序后的数组，其
		每一位 nums[i] == i，如果有重复元素，那么必然会有多个位置出现相同的
		元素，而有些位置则没有与该下标相同的元素，由此我们可以通过对数组排序
		来找到重复元素。

		我们遍历原数组，对每一个 nums[i] 做如下处理：
			1、判断 nums[i] == i ?
				ture：
					i++
				false：
					判断 nums[i] == nums[nums[i]] ?
						true：
							遇到重复元素，返回 nums[i]
						false:
							交换 nums[i] 与 nums[nums[i]]，把 nums[i] 的
							值交换到它排序后该出现的位置。
			2、重复 1 的过程，直至找到重复元素或是遍历结束
	时间复杂度：O(n)
		我们需要遍历数组进行交换排序，对于每一个位置我们最多交换两次，故时间
		复杂度为 O(n)。
	空间复杂度：O(1)
*/
func findRepeatNumber2(nums []int) int {
	for i, v := range nums {
		for v != i {
			if v == nums[v] {
				return v
			}
			nums[i], nums[v] = nums[v], nums[i]
		}
		i++
	}
	return -1
}

/*
	方法三：数组映射元素预处理
	思路：
		已知长度为 n 的数组 nums 里面的所有数字都在 0~n-1 的范围内。

		我们遍历 nums，对每一个数组元素 nums[i] 的值 k 所映射的另一个数组
		元素 nums[k] (即 nums[nums[i]]) 做预处理，预先把映射元素 nums[k]
		设为负数，该负数需要能够还原为原来的数，在遍历处理过程中，如果
		k 小于 0，则把 k 还原为原来的数，继续对 nums[k] 做预处理，如果
		nums[k] 小于 0，说明 nums[k] 已经被预处理过，此时找到了重复元素 k，
		把 k 进行返回即可。
		具体处理如下：
			1、首先取 k = nums[i], 判断 k < 0 ?
				true：
					还原 k，即 k += n，n 是数组长度
				false：
					不做任何处理。
			2、判断 nums[k] < 0 ?
				true：
					找到重复元素，返回 k。重复原因是 nums[k] 已经被预处理过。
				false：
					不做任何处理。
			3、预先对 k 所映射的数组元素 nums[k] 做预处理，让
				nums[k] -= n，n 为数组长度，处理后 nums[k] 会小于 0。
		重复上述步骤，直至找到重复元素或遍历完成。
	时间复杂度：O(n)
		最坏情况下我们需要遍历完整个数组对每一个映射元素进行预处理，最好情况
		下我们只需要处理前面两个元素就能找到重复元素。
	空间复杂度：O(1)
*/
func findRepeatNumber3(nums []int) int {
	n := len(nums)
	for _, k := range nums {
		if k < 0 {
			k += n
		}
		if nums[k] < 0 {
			return k
		}
		nums[k] -= n
	}
	return -1
}

/*
题目二：不修改数组找出重复数字
在一个长度为 n+1 的数组里的所有数字都在 1~n 的范围内，所有数组中至少有一个数
字是重复的，请找出数组中任意一个重复的数字，但不能修改输入的数组。

示例 1：
输入：
[2, 3, 5, 4, 3, 2, 6, 7]
输出：2 或 3
*/
/*
	方法一：数组复制
	思路：
		我们创建一个长度为 n+1 的辅助数组，把原数组的数 nums[i] 按排序顺序
		复制到辅助数组 arr[nums[i]] 中去，复制之前先判断位置 arr[nums[i]]
		是否已经存在数字，存在则说明找到了重复数字 nums[i]，把它返回。
	时间复杂度：O(n)
		我们需要遍历原数组，把每一个元素复制到辅助数组中去，最坏情况下需要
		复制所有数组元素。
	空间复杂度：O(n)
		我们需要创建一个长度为 n+1 的辅助数组来完成题目要求。
*/
func findRepeatNumberII(nums []int) int {
	// 此处的 n 相当于题目中的 n+1
	n := len(nums)
	if n == 0 {
		return -1
	}
	arr := make([]int, n)
	for _, v := range nums {
		if arr[v] == v {
			return v
		}
		arr[v] = v
	}
	return -1
}

/*
	方法二：类二分查找【模板二】
	思路：
		方法一需要使用 O(n) 的辅助空间，我们看看能不能不使用辅助空间来完成。
		首先需要考虑的是数组中为什么会有重复数字？假如没有重复数字，则 1~n
		的范围里只有 n 个数字。由于题目所给数组包含 n+1 个数字，所以数组中
		必然包含了重复数字，如此看来在某范围里面数组数字的个数对解题很重要。

		我们把从 1~n 的数字从中间的数字 m 分为两部分，前面一半为 1~m，后面
		一半为 m+1~n，如果 1~m 的数字的个数超过 m，那么这一半的区间里一定
		包含重复数字；否则重复数字在 m+1~n 的区间里。我们可以依此继续对包含
		重复数字的区间进行重复二分，直至找到一个重复的数字为止。
	时间复杂度：O(nlogn)
		n 是数组元素个数，我们采用二分法进行查找，总从查找次数是 O(logn)，
		而每次查找需要调用 countRange 来计算，每次需要 O(n) 的时间，所以
		总的时间复杂度是 O(nlogn)
	空间复杂度：O(1)
	注：
		次算法不能保证找出所有重复的数字，如不能找出 [2, 3, 5, 4, 3, 2, 6, 7]
		中的重复数字 2，因为 1~2 的范围里有 1 和 2 两个数字，而数字 2 刚好也
		出现了 2 次，此时我们用该算法不能确定是 1、2 个出现一次，还是某个数字
		出现了两次。
*/
func findRepeatNumberII2(nums []int) int {
	// 此处的 n 相当于题目中的 n+1
	n := len(nums)
	if n == 0 {
		return -1
	}
	L, R := 1, n-1
	for L < R {
		mid := L + ((R - L) >> 1)
		// 只计算前半段的数字个数
		cnt := countRange(nums, L, mid)
		// 如果前半段的数字个数大于区间的一半，则说明重复元素在前半段，
		// 否则在后半段
		if cnt > (mid - L + 1) {
			R = mid
		} else {
			L = mid + 1
		}
	}
	// 跳出循环时 L == R，mid = L + 0
	cnt := countRange(nums, L, L)
	if cnt > 1 {
		return L
	}
	return -1
}

// 计算 nums 数组中数字在 [start, end] 区间内的个数
func countRange(nums []int, start, end int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	cnt := 0
	for i := 0; i < n; i++ {
		// 计算处在 [start, end] 区间内的元素个数
		if nums[i] >= start && nums[i] <= end {
			cnt ++
		}
	}
	return cnt
}

func main() {
	nums := []int{2, 3, 5, 4, 3, 2, 6, 7}
	res := findRepeatNumberII2(nums)
	fmt.Println(res)
}
