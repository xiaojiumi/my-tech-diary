# 位运算

**基本原理**

0s 表示一串 0，1s 表示一串 1。

```
x ^ 0s = x      x & 0s = 0      x | 0s = x
x ^ 1s = ~x     x & 1s = x      x | 1s = 1s
x ^ x = 0       x & x = x       x | x = x
```

- 利用 x ^ 1s = ~x 的特点，可以将位级表示翻转；利用 x ^ x = 0 的特点，可以将三个数中重复的两个数去除，只留下另一个数。
- 利用 x & 0s = 0 和 x & 1s = x 的特点，可以实现掩码操作。一个数 num 与 mask：00111100 进行位与操作，只保留 num 中与 mask 的 1 部分相对应的位。
- 利用 x | 0s = x 和 x | 1s = 1s 的特点，可以实现设值操作。一个数 num 与 mask：00111100 进行位或操作，将 num 中与 mask 的 1 部分相对应的位都设置为 1。

位与运算技巧：

- n&(n-1) 去除 n 的位级表示中最低的那一位。例如对于二进制表示 10110100，减去 1 得到 10110011，这两个数相与得到 10110000。
- n&(-n) 得到 n 的位级表示中最低的那一位。-n 得到 n 的反码加 1，对于二进制表示 10110100，-n 得到 01001100，相与得到 00000100。
- n-n&(~n+1) 去除 n 的位级表示中最高的那一位。

移位运算：

- \>> n 为算术右移，相当于除以 2n；
- \>>> n 为无符号右移，左边会补上 0。
- << n 为算术左移，相当于乘以 2n。

** mask 计算**

要获取 111111111，将 0 取反即可，~0。

要得到只有第 i 位为 1 的 mask，将 1 向左移动 i-1 位即可，1<<(i-1) 。例如 1<<4 得到只有第 5 位为 1 的 mask ：00010000。

要得到 1 到 i 位为 1 的 mask，(1<<i)-1 即可，例如将 (1<<4)-1 = 00010000-1 = 00001111。

要得到 1 到 i 位为 0 的 mask，只需将 1 到 i 位为 1 的 mask 取反，即 ~((1<<i)-1)。

**Java 中的位操作**

```
static int Integer.bitCount();           // 统计 1 的数量
static int Integer.highestOneBit();      // 获得最高位
static String toBinaryString(int i);     // 转换为二进制表示的字符串
```

# 1. 统计两个数的二进制表示有多少位不同

1. Hamming Distance (Easy)

[Leetcode](https://leetcode.com/problems/hamming-distance/) / [力扣](https://leetcode-cn.com/problems/hamming-distance/)

```
Input: x = 1, y = 4

Output: 2

Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑

The above arrows point to positions where the corresponding bits are different.
```

对两个数进行异或操作，位级表示不同的那一位为 1，统计有多少个 1 即可。

```
public int hammingDistance(int x, int y) {
    int z = x ^ y;
    int cnt = 0;
    while(z != 0) {
        if ((z & 1) == 1) cnt++; //注意这里&和==的优先级
        z = z >> 1;
    }
    return cnt;
}
```

# 2. 数组中唯一一个不重复的元素

\136. Single Number (Easy)

[Leetcode](https://leetcode.com/problems/single-number/description/) / [力扣](https://leetcode-cn.com/problems/single-number/description/)

```
Input: [4,1,2,1,2]
Output: 4
```

两个相同的数异或的结果为 0，对所有数进行异或操作，最后的结果就是单独出现的那个数。

```
public int singleNumber(int[] nums) {
    int ret = 0;
    for (int n : nums) ret = ret ^ n;
    return ret;
}
```

方法二：

```java
class Solution {
    public int singleNumber(int[] nums) {
        HashMap<Integer,Integer> map=new HashMap<>();
        for(Integer num:nums){
            Integer count=map.get(num);  //这里必须要用Integer 否则不能和null进行比较
            count=count==null?1:++count;
            map.put(num,count);
        }
        for(Integer i:map.keySet()){
            Integer count=map.get(i);
            if(count==1)return i; 
        }
        return -1;
    }
}
```

# 3. 找出数组中缺失的那个数

\268. Missing Number (Easy)

[Leetcode](https://leetcode.com/problems/missing-number/description/) / [力扣](https://leetcode-cn.com/problems/missing-number/description/)

```
Input: [3,0,1]
Output: 2
```

题目描述：数组元素在 0-n 之间，但是有一个数是缺失的，要求找到这个缺失的数。

方法三：位运算
分析

由于异或运算（XOR）满足结合律，并且对一个数进行两次完全相同的异或运算会得到原来的数，因此我们可以通过异或运算找到缺失的数字。

算法

我们知道数组中有 nn 个数，并且缺失的数在 [0..n][0..n] 中。因此我们可以先得到 [0..n][0..n] 的异或值，再将结果对数组中的每一个数进行一次异或运算。未缺失的数在 [0..n][0..n] 和数组中各出现一次，因此异或后得到 0。而缺失的数字只在 [0..n][0..n] 中出现了一次，在数组中没有出现，因此最终的异或结果即为这个缺失的数字。

```
public int missingNumber(int[] nums) {
    int ret = 0;
    for (int i = 0; i < nums.length; i++) {
        ret = ret ^ i ^ nums[i];
    }
    return ret ^ nums.length;
}
```

方法四：数学
分析

我们可以用 高斯求和公式 求出 [0..n][0..n] 的和，减去数组中所有数的和，就得到了缺失的数字。


算法

我们在线性时间内可以求出数组中所有数的和，并在常数时间内求出前 n+1个自然数（包括 0）的和，将后者减去前者，就得到了缺失的数字。

```java
class Solution {
    public int missingNumber(int[] nums) {
        int expectedSum = nums.length*(nums.length + 1)/2;
        int actualSum = 0;
        for (int num : nums) actualSum += num;
        return expectedSum - actualSum;
    }
}

```

# 5. 翻转一个数的比特位

\190. Reverse Bits (Easy)

[Leetcode](https://leetcode.com/problems/reverse-bits/description/) / [力扣](https://leetcode-cn.com/problems/reverse-bits/description/)

```
public int reverseBits(int n) {
    int ret = 0;
    for (int i = 0; i < 32; i++) {
        ret <<= 1;
        ret |= (n & 1);
        n >>>= 1;
    }
    return ret;
}
```

# 6. 不用额外变量交换两个整数

[程序员代码面试指南 ：P317](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode 题解 - 位运算.md#)

```
a = a ^ b;
b = a ^ b;
a = a ^ b;
```

# 7. 判断一个数是不是 2 的 n 次方

\231. Power of Two (Easy)

[Leetcode](https://leetcode.com/problems/power-of-two/description/) / [力扣](https://leetcode-cn.com/problems/power-of-two/description/)

二进制表示只有一个 1 存在。

```
public boolean isPowerOfTwo(int n) {
    return n > 0 && Integer.bitCount(n) == 1;
}
```

利用 1000 & 0111 == 0 这种性质，得到以下解法：

```
public boolean isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

# 8. 判断一个数是不是 4 的 n 次方

\342. Power of Four (Easy)

[Leetcode](https://leetcode.com/problems/power-of-four/) / [力扣](https://leetcode-cn.com/problems/power-of-four/)

```java
class Solution {
    public boolean isPowerOfFour(int num) {
        if(num<0)return false;
        while(num>0&&((num%4)==0)){
            num>>=2;
        }
        return num==1;
    }
}
```

也可以使用正则表达式进行匹配。

```
public boolean isPowerOfFour(int num) {
    return Integer.toString(num, 4).matches("10*");
}
```

# 9. 判断一个数的位级表示是否不会出现连续的 0 和 1

\693. Binary Number with Alternating Bits (Easy)

[Leetcode](https://leetcode.com/problems/binary-number-with-alternating-bits/description/) / [力扣](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/description/)

```
Input: 10
Output: True
Explanation:
The binary representation of 10 is: 1010.

Input: 11
Output: False
Explanation:
The binary representation of 11 is: 1011.
```

对于 1010 这种位级表示的数，把它向右移动 1 位得到 101，这两个数每个位都不同，因此异或得到的结果为 1111。

```
public boolean hasAlternatingBits(int n) {
    int a = (n ^ (n >> 1));
    return (a & (a + 1)) == 0;
}
```

# 10. 求一个数的补码

\476. Number Complement (Easy)

[Leetcode](https://leetcode.com/problems/number-complement/description/) / [力扣](https://leetcode-cn.com/problems/number-complement/description/)

```
Input: 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
```

题目描述：不考虑二进制表示中的首 0 部分。

```java
//举个例子，101（5)*111=010
//所以我们只要拿到一个位数和num一样切所有位都为1的数就好了

class Solution {
    public int findComplement(int num) {
        int tmp = num;
        int num2 = 1;
        while(tmp>0){
            num2<<=1;
            tmp>>=1;
        }
        num2-=1; //num2每次循环是 1，10，100，1000，10000，100000，100000 -1后得到111...
        return num^num2;
            
    }
}


```

# 11. 实现整数的加法

\371. Sum of Two Integers (Easy)

[Leetcode](https://leetcode.com/problems/sum-of-two-integers/description/) / [力扣](https://leetcode-cn.com/problems/sum-of-two-integers/description/)

利用位操作实现加法
首先看十进制是如何做的： 5+7=12，三步走

第一步：相加各位的值，不算进位，得到2。
第二步：计算进位值，得到10. 如果这一步的进位值为0，那么第一步得到的值就是最终结果。
第三步：重复上述两步，只是相加的值变成上述两步的得到的结果2和10，得到12。
同样我们可以用三步走的方式计算二进制值相加： 5---101，7---111

第一步：相加各位的值，不算进位，得到010，二进制每位相加就相当于各位做异或操作，101^111。
第二步：计算进位值，得到1010，相当于各位进行与操作得到101，再向左移一位得到1010，(101&111)<<1。
第三步重复上述两步，各位相加 010^1010=1000，进位值为100=(010 & 1010)<<1。
继续重复上述两步：1000^100 = 1100，进位值为0，跳出循环，1100为最终结果。
结束条件：进位为0，即a为最终的求和结果。



```java
class Solution {
    public int getSum(int a, int b) {
        while(b != 0){
            int temp = a ^ b;
            b = (a & b) << 1;
            a = temp;
        }
        return a;
    }
}
```

# 12. 字符串数组最大乘积

\318. Maximum Product of Word Lengths (Medium)

[Leetcode](https://leetcode.com/problems/maximum-product-of-word-lengths/description/) / [力扣](https://leetcode-cn.com/problems/maximum-product-of-word-lengths/description/)

```
Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
Return 16
The two words can be "abcw", "xtfn".
```

题目描述：字符串数组的字符串只含有小写字符。求解字符串数组中两个字符串长度的最大乘积，要求这两个字符串不能含有相同字符。

本题主要问题是判断两个字符串是否含相同字符，由于字符串只含有小写字符，总共 26 位，因此可以用一个 32 位的整数来存储每个字符是否出现过。

```
public int maxProduct(String[] words) {
    int n = words.length;
    int[] val = new int[n];
    for (int i = 0; i < n; i++) {
        for (char c : words[i].toCharArray()) {
            val[i] |= 1 << (c - 'a');
        }
    }
    int ret = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((val[i] & val[j]) == 0) {
                ret = Math.max(ret, words[i].length() * words[j].length());
            }
        }
    }
    return ret;
}
```

# 13. 统计从 0 ~ n 每个数的二进制表示中 1 的个数

\338. Counting Bits (Medium)

[Leetcode](https://leetcode.com/problems/counting-bits/description/) / [力扣](https://leetcode-cn.com/problems/counting-bits/description/)

对于数字 6(110)，它可以看成是 4(100) 再加一个 2(10)，因此 dp[i] = dp[i&(i-1)] + 1;

```
public int[] countBits(int num) {
    int[] ret = new int[num + 1];
    for(int i = 1; i <= num; i++){
        ret[i] = ret[i&(i-1)] + 1;
    }
    return ret;
}
```