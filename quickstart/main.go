// 算法快速入门
package main

import (
	"fmt"
)

func main() {
	strStrTest()
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