// 算法快速入门
package main

import (
	"fmt"
)

func main() {
	// strStrTest()
	subsetsTest()
}

/* 
1、实现 strStr() 函数。
	给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

	示例 1:
	输入: haystack = "hello", needle = "ll"
	输出: 2

	示例 2:
	输入: haystack = "aaaaa", needle = "bba"
	输出: -1

	说明:
	当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
	对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符。

	来源：力扣（LeetCode）
	链接：https://leetcode-cn.com/problems/implement-strstr
	著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

注意：
	在leetcode上提交代码时要把注释去掉，注释也会耗费内存
*/
func strStr(haystack string, needle string) int {
	// 多次使用到的 长度 抽成一个变量保存，避免len()函数多次计算
	needleLen := len(needle)
	if needleLen == 0 {
		return 0
	}

	haystackLen := len(haystack)
	// 用到的局部变量定义到循环外部，避免多次定义带来的性能损耗
	var i, j int
	// 没必要遍历完整个 haystack，由内部循环去对比
	// 不 +1 会在两个字符串相同时错失判断
	for i = 0; i < haystackLen - needleLen + 1; i ++ {
		for j = 0; j < needleLen; j ++ {
			if haystack[i + j] != needle[j] {
				break
			}
		}
		// 判断长度是否相等，如在中间跳出时 j 必定小于 needleLen
		if j == needleLen {
			return i
		}
	}

	return -1
}

// 测试
func strStrTest() {
	haystack := "hello"
	needle := "ll"
	fmt.Println(strStr(haystack, needle))

	haystack = "aa"
	needle = "aa"
	fmt.Println(strStr(haystack, needle))
}

/* 
	2、给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

		说明：解集不能包含重复的子集。

		示例:
		输入: nums = [1,2,3]
		输出:
		[
		[3],
		[1],
		[2],
		[1,2,3],
		[1,3],
		[2,3],
		[1,2],
		[]
		]

		来源：力扣（LeetCode）
		链接：https://leetcode-cn.com/problems/subsets
		著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
	
	思路：
		这是一个典型的应用回溯法的题目，简单来说就是穷尽所有可能性，算法模板如下
			result = []
			func backtrack(选择列表,路径):
				if 满足结束条件:
					result.add(路径)
					return
				for 选择 in 选择列表:
					做选择
					backtrack(选择列表,路径)
					撤销选择
		通过不停的选择，撤销选择，来穷尽所有可能性，最后将满足条件的结果返回
*/
func subsets(nums []int) [][]int {
	numsLen := len(nums)
	if numsLen == 0 {
		return [][]int{}
	}
	
	// 计算结果集的元素个数以便创建切片，避免扩容
	// 由数学可知一个有n个元素的集合，其子集个数为 2^n 个
	resLen := 1
	for i := 0; i < numsLen; i++ {
		resLen *= 2
	}

	// 保存最终结果
	result := make([][]int, 0, resLen)
	// 保存中间结果
	list := make([]int, 0)
	backtrack(nums, 0, list, &result)
	return result
}

// nums 给定的集合
// pos 下次添加到集合中的元素位置索引
// list 临时结果集合(每次需要复制保存)
// result 最终结果，必须是指针类型，否则append中间结果的时候会丢失result
func backtrack(nums []int, pos int, list []int, result *[][]int) {
    // 把临时结果复制出来保存到最终结果
    ans := make([]int, len(list))
    copy(ans, list)
	*result = append(*result, ans)
	
    // 选择、处理结果、再撤销选择
    for i := pos; i < len(nums); i++ {
        list = append(list, nums[i])
        backtrack(nums, i+1, list, result)
        list = list[0 : len(list)-1]
    }
}

// 测试
func subsetsTest() {
	nums := []int{1,2,3}
	fmt.Println(subsets(nums))
}
