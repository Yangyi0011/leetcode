package main

import (
	"fmt"
	"sort"
	"math"
	"strconv"
)

/* 
	golang 使用技巧
*/
func main() {
	// stackFunc()
	// queueFunc()
	// mapFunc()
	// sortFunc()
	// mathFunc()
	// copyFunc()
	// makeFunc() 
	typeConversion()
}

// 栈
// go 可以用切片来模拟栈操作
func stackFunc() {

	// 创建栈
	stack := make([]int, 0)
	
	// 压栈
	push := func(val int) {
		stack = append(stack, val)
	}

	// 栈是否为空
	isEmpty := func() bool {
		return len(stack) == 0
	}

	// 出栈
	pop := func() int {
		if isEmpty() {
			fmt.Println("栈为空！")
			return 0
		}

		res := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]
		return res
	}

	// 测试
	for {
		fmt.Printf("1、压栈\n2、出栈\n0、退出\n请选择：")
		var flag int
		fmt.Scanln(&flag)
		switch flag {
			case 0:{
				return
			}
			case 1: {
				fmt.Printf("请输入要压栈的数：")
				var val int
				fmt.Scanln(&val)
				push(val)
				fmt.Println("当前栈：", stack)
			}
			case 2: {
				val := pop()
				fmt.Println("出栈元素：", val)
				fmt.Println("当前栈：", stack)
			}
			default: {
				fmt.Println("输入有误，请重新输入")
			}
		}
		// 每轮操作完成空一行，美观
		fmt.Println()
	}
}

// 队列
// go 可以用切片来模拟队列操作
func queueFunc() {
	// 创建一个队列
	queue := make([]int, 0)

	// 入队
	push := func(val int) {
		queue = append(queue, val)
	}

	// 队列是否为空
	isEmpty := func() bool {
		return len(queue) == 0
	}

	// 出队
	pop := func() int {
		if isEmpty() {
			fmt.Println("队列为空！")
			return 0
		}

		val := queue[0]
		queue = queue[1:]
		return val
	}

	// 测试
	for {
		fmt.Printf("1、入队\n2、出队\n0、退出\n请选择：")
		var flag int
		fmt.Scanln(&flag)
		switch flag {
			case 0:{
				return
			}
			case 1: {
				fmt.Printf("请输入要入队的数：")
				var val int
				fmt.Scanln(&val)
				push(val)
				fmt.Println("当前队列：", queue)
			}
			case 2: {
				val := pop()
				fmt.Println("出队元素：", val)
				fmt.Println("当前队列：", queue)
			}
			default: {
				fmt.Println("输入有误，请重新输入")
			}
		}
		// 每轮操作完成空一行，美观
		fmt.Println()
	}
}

// 字典
// go 提供 map 来实现字典
// 注意点：
// 		map 键需要可比较，不能为 slice、map、function
// 		map 值都有默认值，可以直接操作默认值，如：m[age]++ 值由 0 变为 1
// 		比较两个 map 需要遍历，其中的 kv 是否相同，因为有默认值关系，所以需要检查 val 和 ok 两个值
func mapFunc() {
	// 创建字典
	mp := make(map[string] int)

	// 添加字典元素
	put := func(k string, v int) {
		mp[k] = v
	}

	// 获取字典元素
	get := func(k string) int {
		return mp[k]
	}

	// 删除字典元素
	remove := func(k string) {
		delete(mp, k)
	}

	// 遍历字典
	print := func() {
		for k, v := range mp {
			fmt.Printf("%v:%v\n", k, v)
		}
	}

	// 测试
	for {
		fmt.Printf("1、添加\n2、获取\n3、删除\n4、遍历\n0、退出\n请选择：")
		var flag int
		fmt.Scanln(&flag)
		switch flag {
			case 0:{
				return
			}
			case 1: {
				fmt.Printf("请输入要添加元素的key：")
				var k string
				fmt.Scanln(&k)
				fmt.Printf("请输入要添加元素的value：")
				var v int
				fmt.Scanln(&v)
				put(k, v)
				fmt.Println("当前map：", mp)
			}
			case 2: {
				fmt.Printf("请输入要获取元素的key：")
				var k string
				fmt.Scanln(&k)
				v := get(k)
				fmt.Printf("获取结果：%v:%v\n", k, v)
				fmt.Println("当前map：", mp)
			}
			case 3: {
				fmt.Printf("请输入要删除元素的key：")
				var k string
				fmt.Scanln(&k)
				remove(k)
				fmt.Println("当前map：", mp)
			}
			case 4: {
				print()
			}
			default: {
				fmt.Println("输入有误，请重新输入")
			}
		}
		// 每轮操作完成空一行，美观
		fmt.Println()
	}
}

// 标准库-sort
// 需要导入 sort 包
func sortFunc() {
	// int 排序
	ints := []int{2,1,4,5,7,3,9,8,6}
	sort.Ints(ints)
	fmt.Println("ints排序后：", ints)

	// 字符串 排序
	strs := []string{"E", "C", "B", "F", "g", "A", "D"}
	sort.Strings(strs)
	fmt.Println("strs排序后：", strs)

	// 自定义排序
	slice := []int{2,1,4,5,7,3,9,8,6}
	sort.Slice(slice, func(i, j int) bool {
		// 逆序
		return slice[i] > slice[j]
	})
	fmt.Println("slice排序后：", slice)
}

// 标准库-math
// 需要导入 math 包
func mathFunc() {
	// int32 最大值：1<<31-1  即 (2^31)-1
	maxInt32 := math.MaxInt32
	// int32 最小值：-1<<31  即 -(2^31)
	minInt32 := math.MinInt32

	// int64 最大值：1<<63-1 即 (2^63)-1
	maxInt64 := math.MaxInt64
	// int64 最小值：-1<<63  即 -(2^63)
	minInt64 := math.MinInt64

	// maxInt32: 2147483647
	fmt.Println("maxInt32:", maxInt32)		
	// minInt32: -2147483648
	fmt.Println("minInt32:", minInt32)		
	// maxInt64: 9223372036854775807
	fmt.Println("maxInt64:", maxInt64)		
	// minInt64: -9223372036854775808
	fmt.Println("minInt64:", minInt64)		
}

// 内建函数-copy
func copyFunc() {
	// 删除 a[i]，可以用 copy 将 i+1 到末尾的值覆盖到 i，然后末尾 -1
	a := []int{1,2,3,4,5}

	// 想删除a[2]
	i := 2
	copy(a[i:], a[i+1:])
	a = a[0:len(a) - 1]

	// [1 2 4 5]
	fmt.Println(a)
}

// 内建函数-make
// 注意点：
// 		切片底层是一个结构体（struct），包含3个要素：
//			1、指向数据存放数组的指针-ptr
// 			2、当前切片已使用长度-len
// 			3、切片总容量-cap
// 		make([]int, 0, 10) 会分配一个长度为0，容量为10的切片，make([]int, 5) 会分配一个长度为5，容量为5的切片。
// 		在使用 append() 给切片追加元素时，若是切片元素超过切片的总容量，则会触发切片的自动扩容机制。
// 		切片的自动扩容机制：
// 			当需要的容量超过原切片容量的两倍时，会使用需要的容量作为新容量。
//			当原切片长度小于1024时，新切片的容量会直接翻倍。
// 			当原切片的容量大于等于1024时，会反复地增加25%，直到新容量超过所需要的容量。
func makeFunc() {
	// make 创建有长度的切片，则通过索引赋值
	a := make([]int, 2)
	a[0] = 1
	a[1] = 2
	// [1 2]
	fmt.Println(a)

	// 创建容量为0的切片，则通过 append() 追加值
	s := make([]int, 0)
	s = append(s, 1)
	s = append(s, 2)
	// [1 2]
	fmt.Println(s)
}

// 常用类型转换技巧
func typeConversion() {
	s := "12345"

	// byte 转数字，s[0] 是byte类型
	num := int(s[0] - '0')	// 1
	fmt.Printf("num类型：%T，num值：%v\n", num, num)

	// int 转 byte
	b := byte(num + '0')	// '1'
	fmt.Printf("b类型：%T，b值：%c\n", b, b)

	// byte 转 string
	str := string(s[0])		// "1"
	fmt.Printf("str类型：%T，str值：%v\n", str, str)

	// 字符串转数字
	ints, _ := strconv.Atoi(s)
	fmt.Printf("ints类型：%T， ints值：%v\n", ints, ints)

	// 数字转字符串
	strs := strconv.Itoa(ints)
	fmt.Printf("strs类型：%T，strs值：%v\n", strs, strs)

	/* 
		输出结果：
			num类型：int，num值：1
			b类型：uint8，b值：1
			str类型：string，str值：1
			ints类型：int， ints值：12345
			strs类型：string，strs值：12345
	*/
}