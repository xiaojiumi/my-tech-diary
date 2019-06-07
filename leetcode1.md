- [双指针](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E5%8F%8C%E6%8C%87%E9%92%88.md)
- [排序](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E6%8E%92%E5%BA%8F.md)
- [贪心思想](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E8%B4%AA%E5%BF%83%E6%80%9D%E6%83%B3.md)
- [二分查找](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE.md)
- [分治](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E5%88%86%E6%B2%BB.md)
- [搜索](https://github.com/CyC2018/CS-Notes/blob/master/notes/Leetcode%20%E9%A2%98%E8%A7%A3%20-%20%E6%90%9C%E7%B4%A2.md)



#345. 反转字符串中的元音字母

编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

**示例 1:**

```
输入: "hello"
输出: "holle"
```

**示例 2:**

```
输入: "leetcode"
输出: "leotcede"
```

**说明:**
元音字母不包含字母"y"。



```java
class Solution {
    public String reverseVowels(String s) {
      HashSet<Character> vowels = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));
        int i=0,j=s.length()-1;
        char[] result=new char[s.length()];
        while(i<=j){
            char ci=s.charAt(i);
            char cj=s.charAt(j);
        if(!vowels.contains(ci)){
            result[i++]=ci;
        }else if(!vowels.contains(cj)){
            result[j--]=cj;
        }else{
            result[i++]=cj;
            result[j--]=ci;
        }
        }return result.toString();
    
    }
    
}
```



# 347.前K个高频元素

给定一个非空的整数数组，返回其中出现频率前 **k** 高的元素。

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

**说明：**

- 你可以假设给定的 *k* 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度**必须**优于 O(*n* log *n*) , *n* 是数组的大小。

```java
//小根堆
class Solution 
{
    public List<Integer> topKFrequent(int[] nums, int k)
    {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums)
        {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>(new Comparator<Integer>()
        {
            public int compare(Integer a, Integer b)
            {
                return map.get(a) - map.get(b);
            }
        });
        for (int key : map.keySet())
        {
            if (pq.size() < k)
                pq.add(key);
            else if (map.get(key) > map.get(pq.peek()))
            {
                pq.remove();
                pq.add(key);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!pq.isEmpty())
        {
            res.add(pq.remove());
        }
        return res;
    }
}
```
#451给定一个字符串，请将字符串里的字符按照出现的频率降序排列。

**示例 1:**

```
输入:
"tree"

输出:
"eert"

解释:
'e'出现两次，'r'和't'都只出现一次。
因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
```

**示例 2:**

```
输入:
"cccaaa"

输出:
"cccaaa"

解释:
'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
注意"cacaca"是不正确的，因为相同的字母必须放在一起。
```

**示例 3:**

```
输入:
"Aabb"

输出:
"bbAa"

解释:
此外，"bbaA"也是一个有效的答案，但"Aabb"是不正确的。
注意'A'和'a'被认为是两种不同的字符。
```

```java
//利用hash创建数组，索引就是字符，数组值就是出现次数，然后排序找到，时间O(n*logn),击败94%;
class Solution {
    public String frequencySort(String s) {
        int[] hash=new int[256];
		StringBuilder sb=new StringBuilder();
		for(int i=0;i<s.length();i++){
			hash[s.charAt(i)]++;
		}
		int[] hash1=hash.clone();//这个数组是为了找到hash数组排序后对应的下标
		Arrays.sort(hash);//将数组排序
		for(int i=255;i>=0&&hash[i]>0;i--){//为0的部分直接不用管
			for(int j=0;j<256;j++){   //从未排序的里面找到对应值
				if(hash[i]==hash1[j]){
					while(hash1[j]-->0)
						sb.append((char)j);
				}
			}
		}
		return sb.toString();
    }
}

```

#75. Sort Colors (Medium) 

给定一个包含红色、白色和蓝色，一共 *n* 个元素的数组，**原地**对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

**注意:**
不能使用代码库中的排序函数来解决这道题。

**示例:**

```
输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]
```

**进阶：**

- 一个直观的解决方案是使用计数排序的两趟扫描算法。
  首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。

- 你能想出一个仅使用常数空间的一趟扫描算法吗？

  

```java
 public void sortColors(int[] nums) {
     // [0...zero] == 0，初始情况zero为-1，此时区间无效。
   //如果初始值为1,那就是说区间中第一个元素是0，
   //而第一个元素未知，所以这里初始值设为-1；   
     int zero=-1;
        int two=nums.length; // [two...n-1] == 2,同理初始区间是无效区间
        
        for(int  i=0;i<two;){
            if(nums[i]==1){
                i++;
            }else if(nums[i]==2){
                swap(nums,i,--two);
            }else{
                swap(nums,++zero,i++);
            }
        }
        
    }
    
    
     private void swap(int[] nums, int i, int j){
        int t = nums[i];
        nums[i]= nums[j];
        nums[j] = t;
    }
```

我们遍历到第i个元素。第i个元素值为2，我们就将第i个元素的值放到two-1的位置。同时将two-1位置的元素放到i的位置。就是交换这两个位置的元素。因为找出为2的元素增加了一个，所以two要减一。而此时i是不变的，因为交换过来的数据是啥仍然是未知的。同理可以分析出当第i个元素如果是0和1要应该怎么处理。

# [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

**注意:**

1. 可以认为区间的终点总是大于它的起点。
2. 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

**示例 1:**

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
```

**示例 2:**

```
输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
```

**示例 3:**

```
输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```

```java
 public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length < 2) {
            return 0;
        }
        Arrays.sort(intervals, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
        int ans = 0;
        int beforL = intervals[0][0];
        int beforR = intervals[0][1];
        for (int i = 1; i < intervals.length; i++) {

            int[] interval = intervals[i];
            //没有重叠,下一个
            if (interval[0] >= beforR) {
                beforL = interval[0];
                beforR = interval[1];
                continue;
            }

            //重叠,必须在当前和之前中去掉一个
            ans++;
            //如果是[1,4],[2,3]那么就直接下一个,更新before的值
            //如果是[1,4],[2,5]直接去掉当前(即不更新before),因为当前的[2,5]右边5>4,会导致重叠的几率更大
            if (interval[1] < beforR) {
                beforL = interval[0];
                beforR = interval[1];
            }
        }
        return ans;
    }
```



 #[406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

 假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对`(h, k)`表示，其中`h`是这个人的身高，`k`是排在这个人前面且身高大于或等于`h`的人数。 编写一个算法来重建这个队列。

**注意：**
总人数少于1100人。

**示例**

```
输入:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

输出:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

 ```java
if (0 == people.length || 0 == people[0].length)
            return new int[0][0];
         //按照身高降序 K升序排序 
       Arrays.sort(people, (a, b) -> (a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]));
        List<int[]> list = new ArrayList<>();
        //K值定义为 排在h前面且身高大于或等于h的人数 
        //因为从身高降序开始插入，此时所有人身高都大于等于h
        //因此K值即为需要插入的位置
        for (int[] i : people) {
            list.add(i[1], i);
        }
        return list.toArray(new int[list.size()][]);

 ```



# [605. Can Place Flowers (Easy)](https://leetcode.com/problems/can-place-flowers/description/)

```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: True
```

题目描述：flowerbed 数组中 1 表示已经种下了花朵。花朵之间至少需要一个单位的间隔，求解是否能种下 n 朵花。

```java
public boolean canPlaceFlowers(int[] flowerbed, int n) {
    int len = flowerbed.length;
    int cnt = 0;
    for (int i = 0; i < len && cnt < n; i++) {
        if (flowerbed[i] == 1) {
            continue;
        }
        int pre = i == 0 ? 0 : flowerbed[i - 1];
        int next = i == len - 1 ? 0 : flowerbed[i + 1];
        if (pre == 0 && next == 0) {
            cnt++;
            flowerbed[i] = 1;
        }
    }
    return cnt >= n;
}
```



#[392. Is Subsequence (Medium)](https://leetcode.com/problems/is-subsequence/description/)

```
s = "abc", t = "ahbgdc"
Return true.
public boolean isSubsequence(String s, String t) {
    int index = -1;
    for (char c : s.toCharArray()) {
        index = t.indexOf(c, index + 1);
        if (index == -1) {
            return false;
        }
    }
    return true;
}
```


#[665. Non-decreasing Array (Easy)](https://leetcode.com/problems/non-decreasing-array/description/)

```
Input: [4,2,3]
Output: True
Explanation: You could modify the first 4 to 1 to get a non-decreasing array.
```

题目描述：判断一个数组是否能只修改一个数就成为非递减数组。

在出现 nums[i] < nums[i - 1] 时，需要考虑的是应该修改数组的哪个数，使得本次修改能使 i 之前的数组成为非递减数组，并且 **不影响后续的操作** 。优先考虑令 nums[i - 1] = nums[i]，因为如果修改 nums[i] = nums[i - 1] 的话，那么 nums[i] 这个数会变大，就有可能比 nums[i + 1] 大，从而影响了后续操作。还有一个比较特别的情况就是 nums[i] < nums[i - 2]，修改 nums[i - 1] = nums[i] 不能使数组成为非递减数组，只能修改 nums[i] = nums[i - 1]。

```java
public boolean checkPossibility(int[] nums) {
    int cnt = 0;
    for (int i = 1; i < nums.length && cnt < 2; i++) {
        if (nums[i] >= nums[i - 1]) {
            continue;
        }
        cnt++;
        if (i - 2 >= 0 && nums[i - 2] > nums[i]) {
            nums[i] = nums[i - 1];
        } else {
            nums[i - 1] = nums[i];
        }
    }
    return cnt <= 1;
}
```



#[763. Partition Labels (Medium)](https://leetcode.com/problems/partition-labels/description/)

```
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
public List<Integer> partitionLabels(String S) {
    int[] lastIndexsOfChar = new int[26];
    for (int i = 0; i < S.length(); i++) {
        lastIndexsOfChar[char2Index(S.charAt(i))] = i;
    }
    List<Integer> partitions = new ArrayList<>();
    int firstIndex = 0;
    while (firstIndex < S.length()) {
        int lastIndex = firstIndex;
        for (int i = firstIndex; i < S.length() && i <= lastIndex; i++) {
            int index = lastIndexsOfChar[char2Index(S.charAt(i))];
            if (index > lastIndex) {
                lastIndex = index;
            }
        }
        partitions.add(lastIndex - firstIndex + 1);
        firstIndex = lastIndex + 1;
    }
    return partitions;
}

private int char2Index(char c) {
    return c - 'a';
}
```




#[69. Sqrt(x) (Easy)](https://leetcode.com/problems/sqrtx/description/)

```
Input: 4
Output: 2

Input: 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we want to return an integer, the decimal part will be truncated.
```

一个数 x 的开方 sqrt 一定在 0 ~ x 之间，并且满足 sqrt == x / sqrt。可以利用二分查找在 0 ~ x 之间查找 sqrt。

对于 x = 8，它的开方是 2.82842...，最后应该返回 2 而不是 3。在循环条件为 l <= h 并且循环退出时，h 总是比 l 小 1，也就是说 h = 2，l = 3，因此最后的返回值应该为 h 而不是 l。

```
public int mySqrt(int x) {
    if (x <= 1) {
        return x;
    }
    int l = 1, h = x;
    while (l <= h) {
        int mid = l + (h - l) / 2;
        int sqrt = x / mid;
        if (sqrt == mid) {
            return mid;
        } else if (mid > sqrt) {
            h = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    return h;
}
```





#[540. Single Element in a Sorted Array (Medium)](https://leetcode.com/problems/single-element-in-a-sorted-array/description/)

```
Input: [1, 1, 2, 3, 3, 4, 4, 8, 8]
Output: 2
```

题目描述：一个有序数组只有一个数不出现两次，找出这个数。

要求以 O(logN) 时间复杂度进行求解，因此不能遍历数组并进行异或操作来求解，这么做的时间复杂度为 O(N)。

令 index 为 Single Element 在数组中的位置。在 index 之后，数组中原来存在的成对状态被改变。如果 m 为偶数，并且 m + 1 < index，那么 nums[m] == nums[m + 1]；m + 1 >= index，那么 nums[m] != nums[m + 1]。

从上面的规律可以知道，如果 nums[m] == nums[m + 1]，那么 index 所在的数组位置为 [m + 2, h]，此时令 l = m + 2；如果 nums[m] != nums[m + 1]，那么 index 所在的数组位置为 [l, m]，此时令 h = m。

因为 h 的赋值表达式为 h = m，那么循环条件也就只能使用 l < h 这种形式。

```
public int singleNonDuplicate(int[] nums) {
    int l = 0, h = nums.length - 1;
    while (l < h) {
        int m = l + (h - l) / 2;
        if (m % 2 == 1) {
            m--;   // 保证 l/h/m 都在偶数位，使得查找区间大小一直都是奇数
        }
        if (nums[m] == nums[m + 1]) {
            l = m + 2;
        } else {
            h = m;
        }
    }
    return nums[l];
}
```



# [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

给定一个按照升序排列的整数数组 `nums`，和一个目标值 `target`。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 *O*(log *n*) 级别。

如果数组中不存在目标值，返回 `[-1, -1]`。

**示例 1:**

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
```

**示例 2:**

```
输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]
```

```java
public int[] searchRange(int[] nums, int target) {
        int[] res={-1,-1};
        if(nums.length==0)return res;
        int left=0;
        int right=0;
        int l=0,h=nums.length-1;
        while(l<=h){
            int mid=l+(h-l)/2;
            if(nums[mid]==target){
                left=mid;
                right=mid;
                while(left>0&&nums[left]==nums[left-1])left--;
                while(right<nums.length-1&&nums[right]==nums[right+1])right++;
                res[0]=left;
                res[1]=right;
                return res;
            }else if(nums[mid]>target)
                 h=mid-1;
            else
                l=mid+1;
        }
        
        return res;
    }
```





#[241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)

给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 `+`, `-` 以及 `*` 。

**示例 1:**

```
输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2
```

**示例 2:**

```
输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```

【笔记】回溯搜索。该问题牵涉到括号的组合问题，一般使用递归+回溯的思想。主要想法：

其一，递归回溯。可以产生所有的组合方式。

其二，每个小组合方式相当于一个子集，不断的将计算结果返回给上一层。

- 举例：`a + (b - (c * d))`会不断的变成`a + (b - (res1 * res2))`-> `a + (res1 - res2)` -> `res1 + res2`
- 似乎计算结果不需要`for`循环？其实有这种情况，`a + (b - (c * d))`和`a + (b - c) * d))`，这里 `a + res2`，`res2`就可能有多种情况。

介绍一个API：`substring`第一个参数为起始位置，第二个参数为长度。如果不设置长度的话，默认为到最尾部。



```java
public List<Integer> diffWaysToCompute(String input) {
    List<Integer> ways = new ArrayList<>();
    for (int i = 0; i < input.length(); i++) {
        char c = input.charAt(i);
        if (c == '+' || c == '-' || c == '*') {
            List<Integer> left = diffWaysToCompute(input.substring(0, i));
            List<Integer> right = diffWaysToCompute(input.substring(i + 1));
            for (int l : left) {
                for (int r : right) {
                    switch (c) {
                        case '+':
                            ways.add(l + r);
                            break;
                        case '-':
                            ways.add(l - r);
                            break;
                        case '*':
                            ways.add(l * r);
                            break;
                    }
                }
            }
        }
    }
    if (ways.size() == 0) {
        ways.add(Integer.valueOf(input));
    }
    return ways;
}
```





#[95. Unique Binary Search Trees II (Medium)](https://leetcode.com/problems/unique-binary-search-trees-ii/description/)

给定一个数字 n，要求生成所有值为 1...n 的二叉搜索树。

```
Input: 3
Output:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
Explanation:
The above output corresponds to the 5 unique BST's shown below:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
public List<TreeNode> generateTrees(int n) {
    if (n < 1) {
        return new LinkedList<TreeNode>();
    }
    return generateSubtrees(1, n);
}

private List<TreeNode> generateSubtrees(int s, int e) {
    List<TreeNode> res = new LinkedList<TreeNode>();
    if (s > e) {
        res.add(null);
        return res;
    }
    for (int i = s; i <= e; ++i) {
        List<TreeNode> leftSubtrees = generateSubtrees(s, i - 1);
        List<TreeNode> rightSubtrees = generateSubtrees(i + 1, e);
        for (TreeNode left : leftSubtrees) {
            for (TreeNode right : rightSubtrees) {
                TreeNode root = new TreeNode(i);
                root.left = left;
                root.right = right;
                res.add(root);
            }
        }
    }
    return res;
}
```


# [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

给定两个单词（*beginWord* 和 *endWord*）和一个字典，找到从 *beginWord* 到 *endWord* 的最短转换序列的长度。转换需遵循如下规则： 

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

- 如果不存在这样的转换序列，返回 0。
- 所有单词具有相同的长度。
- 所有单词只由小写字母组成。
- 字典中不存在重复的单词。
- 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

**示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

输出: 5

解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。
```

**示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

输出: 0

解释: endWord "cog" 不在字典中，所以无法进行转换。
```

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> wordSet = new HashSet<>(wordList); //替换掉题目中List结构，加速查找
        if (!wordSet.contains(endWord)) return 0; //如果目标顶点不在图中，直接返回0
        HashMap<String, Integer> map = new HashMap<>(); //用来存储已访问的节点，并存储其在路径上的位置，相当于BFS算法中的isVisted功能
        Queue<String> q = new LinkedList<>(); //构建队列，实现广度优先遍历
        q.add(beginWord); //加入源顶点
        map.put(beginWord, 1); //添加源顶点为“已访问”，并记录它在路径的位置
        while (!q.isEmpty()) { //开始遍历队列中的顶点
            String word = q.poll(); //记录现在正在处理的顶点
            int level = map.get(word); //记录现在路径的长度
            for (int i = 0; i < word.length(); i++) {
                char[] wordLetter = word.toCharArray();
                for (char j = 'a'; j <= 'z'; j++) {
                    if (wordLetter[i] == j) continue; 
                    wordLetter[i] = j; //对于每一位字母，分别替换成另外25个字母
                    String check = new String(wordLetter);
                    if (check.equals(endWord)) return map.get(word) + 1; //如果已经抵达目标节点，返回当前路径长度+1
                    if (wordSet.contains(check) && !map.containsKey(check)) { //如果字典中存在邻接节点，且这个邻接节点还未被访问
                        map.put(check, level + 1); //标记这个邻接节点为已访问，记录其在路径上的位置
                        q.add(check); //加入队列，以供广度搜索
                    }
                }
            }
        }
        return 0;
    }
```



# [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

给定一个包含了一些 0 和 1的非空二维数组 `grid` , 一个 **岛屿** 是由四个方向 (水平或垂直) 的 `1` (代表土地) 构成的组合。你可以假设二维矩阵的四个边缘都被水包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0。)

**示例 1:**

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
```

对于上面这个给定矩阵应返回 `6`。注意答案不应该是11，因为岛屿只能包含水平或垂直的四个方向的‘1’。

**示例 2:**

```
[[0,0,0,0,0,0,0,0]]
```

对于上面这个给定的矩阵, 返回 `0`。

**注意:** 给定的矩阵`grid` 的长度和宽度都不超过 50。



```java
 private int m,n;
    private int[][] direction={{0,1},{0,-1},{1,0},{-1,0}};
    
    public int maxAreaOfIsland(int[][] grid) {
        if(grid==null||grid.length==0)
            return 0;
        m=grid.length;
        n=grid[0].length;
        int maxArea=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                maxArea=Math.max(maxArea,dfs(grid,i,j));
            }
        }
        return maxArea;
    }
    
    private int dfs(int[][] grid,int r,int c){
        if(r<0||r>=m||c<0||c>=n||grid[r][c]==0)
            return 0;
        grid[r][c]=0;//标记已经找到的土地
        int area=1;
        for(int[] d:direction){
            area+=dfs(grid,r+d[0],c+d[1]);
        }
        return area;
    }
```



#[547. Friend Circles (Medium)](https://leetcode.com/problems/friend-circles/description/)

```
Input:
[[1,1,0],
 [1,1,0],
 [0,0,1]]

Output: 2

Explanation:The 0th and 1st students are direct friends, so they are in a friend circle.
The 2nd student himself is in a friend circle. So return 2.
```

题目描述：好友关系可以看成是一个无向图，例如第 0 个人与第 1 个人是好友，那么 M[0][1] 和 M[1][0] 的值都为 1。

```
private int n;

public int findCircleNum(int[][] M) {
    n = M.length;
    int circleNum = 0;
    boolean[] hasVisited = new boolean[n];
    for (int i = 0; i < n; i++) {
        if (!hasVisited[i]) {
            dfs(M, i, hasVisited);
            circleNum++;
        }
    }
    return circleNum;
}

private void dfs(int[][] M, int i, boolean[] hasVisited) {
    hasVisited[i] = true;
    for (int k = 0; k < n; k++) {
        if (M[i][k] == 1 && !hasVisited[k]) {
            dfs(M, k, hasVisited);
        }
    }
}
```



#[130. Surrounded Regions (Medium)](https://leetcode.com/problems/surrounded-regions/description/)

```
For example,
X X X X
X O O X
X X O X
X O X X

After running your function, the board should be:
X X X X
X X X X
X X X X
X O X X
```

题目描述：使被 'X' 包围的 'O' 转换为 'X'。

先填充最外侧，剩下的就是里侧了。

```
private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
private int m, n;

public void solve(char[][] board) {
    if (board == null || board.length == 0) {
        return;
    }

    m = board.length;
    n = board[0].length;

    for (int i = 0; i < m; i++) {
        dfs(board, i, 0);//找到边界上的O及与它相连的O并将其标记，其他O必然被X包围。
        dfs(board, i, n - 1);
    }
    for (int i = 0; i < n; i++) {
        dfs(board, 0, i);
        dfs(board, m - 1, i);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 'T') {
                board[i][j] = 'O';
            } else if (board[i][j] == 'O') {
                board[i][j] = 'X';
            }
        }
    }
}

private void dfs(char[][] board, int r, int c) {
    if (r < 0 || r >= m || c < 0 || c >= n || board[r][c] != 'O') {
        return;
    }
    board[r][c] = 'T';
    for (int[] d : direction) {
        dfs(board, r + d[0], c + d[1]);
    }
}
```



#[417. Pacific Atlantic Water Flow (Medium)](https://leetcode.com/problems/pacific-atlantic-water-flow/description/)

```
Given the following 5x5 matrix:

  Pacific ~   ~   ~   ~   ~
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic

Return:
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).
```

左边和上边是太平洋，右边和下边是大西洋，内部的数字代表海拔，海拔高的地方的水能够流到低的地方，求解水能够流到太平洋和大西洋的所有位置。

```
private int m, n;
private int[][] matrix;
private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

public List<int[]> pacificAtlantic(int[][] matrix) {
    List<int[]> ret = new ArrayList<>();
    if (matrix == null || matrix.length == 0) {
        return ret;
    }

    m = matrix.length;
    n = matrix[0].length;
    this.matrix = matrix;
    boolean[][] canReachP = new boolean[m][n];
    boolean[][] canReachA = new boolean[m][n];

    for (int i = 0; i < m; i++) {
        dfs(i, 0, canReachP);//找到能从左侧这边进入太平洋的所有点
        dfs(i, n - 1, canReachA);
    }
    for (int i = 0; i < n; i++) {
        dfs(0, i, canReachP);
        dfs(m - 1, i, canReachA);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (canReachP[i][j] && canReachA[i][j]) {
                ret.add(new int[]{i, j});  //注意这里add时的格式
            }
        }
    }

    return ret;
}

private void dfs(int r, int c, boolean[][] canReach) {
    if (canReach[r][c]) {
        return;
    }
    canReach[r][c] = true;
    for (int[] d : direction) {
        int nextR = d[0] + r;
        int nextC = d[1] + c;
        if (nextR < 0 || nextR >= m || nextC < 0 || nextC >= n   
                || matrix[r][c] > matrix[nextR][nextC]) {	//注意这次要将下一次的坐标与边界值比较

            continue;
        }
        dfs(nextR, nextC, canReach);
    }
}
```


#[17. Letter Combinations of a Phone Number (Medium)](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/)

[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/9823768c-212b-4b1a-b69a-b3f59e07b977.jpg)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/9823768c-212b-4b1a-b69a-b3f59e07b977.jpg)

 

```
Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
private static final String[] KEYS = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

public List<String> letterCombinations(String digits) {
    List<String> combinations = new ArrayList<>();
    if (digits == null || digits.length() == 0) {
        return combinations;
    }
    doCombination(new StringBuilder(), combinations, digits);
    return combinations;
}

private void doCombination(StringBuilder prefix, List<String> combinations, final String digits) {
    if (prefix.length() == digits.length()) {
        combinations.add(prefix.toString());
        return;
    }
    int curDigits = digits.charAt(prefix.length()) - '0';
    String letters = KEYS[curDigits];
    for (char c : letters.toCharArray()) {
        prefix.append(c);                         // 添加
        doCombination(prefix, combinations, digits);
        prefix.deleteCharAt(prefix.length() - 1); // 删除
    }
}
```



#[93. Restore IP Addresses(Medium)](https://leetcode.com/problems/restore-ip-addresses/description/)

```
Given "25525511135",
return ["255.255.11.135", "255.255.111.35"].
public List<String> restoreIpAddresses(String s) {
    List<String> addresses = new ArrayList<>();
    StringBuilder tempAddress = new StringBuilder();
    doRestore(0, tempAddress, addresses, s);
    return addresses;
}

private void doRestore(int k, StringBuilder tempAddress, List<String> addresses, String s) {
    if (k == 4 || s.length() == 0) {
        if (k == 4 && s.length() == 0) {
            addresses.add(tempAddress.toString());
        }
        return;
    }
    for (int i = 0; i < s.length() && i <= 2; i++) {
        if (i != 0 && s.charAt(0) == '0') {
            break;     //如果出现00或000，break
        }
        String part = s.substring(0, i + 1);
        if (Integer.valueOf(part) <= 255) {
            if (tempAddress.length() != 0) {
                part = "." + part;
            }
            tempAddress.append(part);
            doRestore(k + 1, tempAddress, addresses, s.substring(i + 1));
            tempAddress.delete(tempAddress.length() - part.length(), tempAddress.length());
            //delete(起始坐标，结束坐标)，将最后一次保存的删除，进行下次回溯
        }
    }
}
```



#[79. Word Search (Medium)](https://leetcode.com/problems/word-search/description/)

```
For example,
Given board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.
private final static int[][] direction = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
private int m;
private int n;

public boolean exist(char[][] board, String word) {
    if (word == null || word.length() == 0) {
        return true;
    }
    if (board == null || board.length == 0 || board[0].length == 0) {
        return false;
    }

    m = board.length;
    n = board[0].length;
    boolean[][] hasVisited = new boolean[m][n];

    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            if (backtracking(0, r, c, hasVisited, board, word)) {
                return true;
            }
        }
    }

    return false;
}

private boolean backtracking(int curLen, int r, int c, boolean[][] visited, final char[][] board, final String word) {
    if (curLen == word.length()) {
        return true;
    }
    if (r < 0 || r >= m || c < 0 || c >= n
            || board[r][c] != word.charAt(curLen) || visited[r][c]) {

        return false;
    }

    visited[r][c] = true;

    for (int[] d : direction) {
        if (backtracking(curLen + 1, r + d[0], c + d[1], visited, board, word)) {
            return true;
        }
    }

    visited[r][c] = false;

    return false;
}
```



#[257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明:** 叶子节点是指没有子节点的节点。

**示例:**

```
输入:

   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```

```java
public List<String> binaryTreePaths(TreeNode root) {
        List<String> res=new ArrayList<>();
        if(root==null)
            return res;
        String temp="";
        dfs(res,temp,root);
        return res;
    }
    
    private void dfs(List<String> res,String temp,TreeNode root){
        if(root==null){
            return;
        }
        temp+=root.val;
        if(root.left==null&&root.right==null){
            res.add(temp);
            return;
        }else{
            dfs(res,temp+"->",root.left);
             dfs(res,temp+"->",root.right);
        }
    }
```



# [46. 全排列](https://leetcode-cn.com/problems/permutations/)

给定一个**没有重复**数字的序列，返回其所有可能的全排列。

**示例:**

```
输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

```java
 public List<List<Integer>> permute(int[] nums) {
        List<Integer> per=new ArrayList<>();
        List<List<Integer>> perList=new ArrayList<>();
        boolean[] boo=new boolean[nums.length];
        backtracking(nums,per,perList,boo);
        return perList;
    }
    
    private void backtracking(int[] nums,List<Integer> per,List<List<Integer>> perList,boolean[] boo){
        if(per.size()==nums.length){
             perList.add(new ArrayList<>(per));
             return;
        }
           
        for(int i=0;i<nums.length;i++){
            if(boo[i])
                continue;
            boo[i]=true;
            per.add(nums[i]);
            backtracking(nums,per,perList,boo);
            per.remove(per.size()-1);
            boo[i]=false;
        }
    }
```



#[47. Permutations II (Medium)](https://leetcode.com/problems/permutations-ii/description/)

```
[1,1,2] have the following unique permutations:
[[1,1,2], [1,2,1], [2,1,1]]
```

数组元素可能含有相同的元素，进行排列时就有可能出现重复的排列，要求重复的排列只返回一个。

在实现上，和 Permutations 不同的是要先排序，然后在添加一个元素时，判断这个元素是否等于前一个元素，如果等于，并且前一个元素还未访问，那么就跳过这个元素。

```
public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> permutes = new ArrayList<>();
    List<Integer> permuteList = new ArrayList<>();
    Arrays.sort(nums);  // 排序
    boolean[] hasVisited = new boolean[nums.length];
    backtracking(permuteList, permutes, hasVisited, nums);
    return permutes;
}

private void backtracking(List<Integer> permuteList, List<List<Integer>> permutes, boolean[] visited, final int[] nums) {
    if (permuteList.size() == nums.length) {
        permutes.add(new ArrayList<>(permuteList));
        return;
    }

    for (int i = 0; i < visited.length; i++) {
        if (i != 0 && nums[i] == nums[i - 1] && !visited[i - 1]) {
            continue;  // 防止重复
        }
        if (visited[i]){
            continue;
        }
        visited[i] = true;
        permuteList.add(nums[i]);
        backtracking(permuteList, permutes, visited, nums);
        permuteList.remove(permuteList.size() - 1);
        visited[i] = false;
    }
}
```



# [77. 组合](https://leetcode-cn.com/problems/combinations/)

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

**示例:**

```
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

思路
标签：回溯与剪枝
n表示范围为1...n，balance表示剩余空间，start表示开始位置，list为回溯列表
判断balance == 0，如果为0则代表list中已经存入k个数，拷贝list存入结果ans中
如果不为0，从start位置开始递归调用，现将当前位置数据加入list中，并进入下一层，等待返回后将本层加入的数据移除，本质就是树的构造过程
其中循环结束条件默认为最大值到n，这里可以优化进行剪枝，比如n=4，k=3时，如果列表从start=3也就是[3]开始，那么该组合一定不存在，因为至少要k=3个数据，所以剪枝临界点为n-balance+1
代码

```java
class Solution {
    private List<List<Integer>> ans = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        getCombine(n, k, 1, new ArrayList<>());
        return ans;
    }
    
    public void getCombine(int n, int k, int start, List<Integer> list) {
        if(k == 0) {
            ans.add(new ArrayList<>(list));
            return;
        }
        for(int i = start;i <= n - k + 1;i++) {
            list.add(i);
            getCombine(n, k - 1, i+1, list);
            list.remove(list.size() - 1);
        }
    }
}
```

#[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

给定一个**无重复元素**的数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的数字可以无限制重复被选取。

**说明：**

- 所有数字（包括 `target`）都是正整数。
- 解集不能包含重复的组合。 

**示例 1:**

```
输入: candidates = [2,3,6,7], target = 7,
所求解集为:
[
  [7],
  [2,2,3]
]
```

**示例 2:**

```
输入: candidates = [2,3,5], target = 8,
所求解集为:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```

```java
private List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        
        backtracking(candidates,target,new ArrayList<>(),0);
        
        return res;
    }
    
    private void backtracking(int[] candidates, int target,List<Integer> list,int index){
        if(target==0){
            res.add(new ArrayList<>(list));
            return;
        }
        for(int i=index;i<candidates.length;i++){
            if(candidates[i]<=target){  //此处必须判定，否则会导致栈溢出
                list.add(candidates[i]);
            backtracking(candidates,target-candidates[i],list,i); //此处要将坐标传递，否则会产生重复度的结果
            list.remove(list.size()-1);
            }
            
        }
    }
```



# [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

给定一个可能包含重复元素的整数数组 **\*nums***，返回该数组所有可能的子集（幂集）。

**说明：**解集不能包含重复的子集。

**示例:**

```
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

```java
 public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
          List<List<Integer>> subsets = new ArrayList<>();
      List<Integer> tempSubset = new ArrayList<>();
        for (int size = 0; size <= nums.length; size++) {
            backtracking(subsets,nums,tempSubset,size,0);
        }
        
        return subsets;
    }
    
    private void backtracking(List<List<Integer>> subsets,int[] nums,List<Integer> tempSubset,int size,int index){
        if (tempSubset.size() == size) {
        subsets.add(new ArrayList<>(tempSubset));
        return;
    }
        
        for(int i=index;i<nums.length;i++){
            if(i>index&&nums[i]==nums[i-1]) //剩余的数组中跳过与当前元素相同的
                continue;
            tempSubset.add(nums[i]);
             backtracking(subsets,nums,tempSubset,size,i+1);
            tempSubset.remove(tempSubset.size()-1);
        }
    }
```



#[131. Palindrome Partitioning (Medium)](https://leetcode.com/problems/palindrome-partitioning/description/)

```
For example, given s = "aab",
Return

[
  ["aa","b"],
  ["a","a","b"]
]
public List<List<String>> partition(String s) {
    List<List<String>> partitions = new ArrayList<>();
    List<String> tempPartition = new ArrayList<>();
    doPartition(s, partitions, tempPartition);
    return partitions;
}

private void doPartition(String s, List<List<String>> partitions, List<String> tempPartition) {
    if (s.length() == 0) {
        partitions.add(new ArrayList<>(tempPartition));
        return;
    }
    for (int i = 0; i < s.length(); i++) {
        if (isPalindrome(s, 0, i)) {
            tempPartition.add(s.substring(0, i + 1));
            doPartition(s.substring(i + 1), partitions, tempPartition);
            tempPartition.remove(tempPartition.size() - 1);
        }
    }
}

private boolean isPalindrome(String s, int begin, int end) {
    while (begin < end) {
        if (s.charAt(begin++) != s.charAt(end--)) {
            return false;
        }
    }
    return true;
}
```


# [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

编写一个程序，通过已填充的空格来解决数独问题。

一个数独的解法需**遵循如下规则**：

1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。

空白格用 `'.'` 表示。

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

一个数独。

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Sudoku-by-L2G-20050714_solution.svg/250px-Sudoku-by-L2G-20050714_solution.svg.png)

答案被标成红色。

**Note:**

- 给定的数独序列只包含数字 `1-9` 和字符 `'.'` 。
- 你可以假设给定的数独只有唯一解。
- 给定数独永远是 `9x9` 形式的。



```java
private boolean[][] rowsUsed = new boolean[9][10];
private boolean[][] colsUsed = new boolean[9][10];
private boolean[][] cubesUsed = new boolean[9][10];
private char[][] board;

public void solveSudoku(char[][] board) {
    this.board = board;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) {
            if (board[i][j] == '.') {
                continue;
            }
            int num = board[i][j] - '0';
            rowsUsed[i][num] = true;
            colsUsed[j][num] = true;
            cubesUsed[cubeNum(i, j)][num] = true;
        }
        backtracking(0, 0);
}

private boolean backtracking(int row, int col) {
    while (row < 9 && board[row][col] != '.') {
        row = col == 8 ? row + 1 : row;
        col = col == 8 ? 0 : col + 1;
    }
    if (row == 9) {
        return true;
    }
    for (int num = 1; num <= 9; num++) {
        if (rowsUsed[row][num] || colsUsed[col][num] || cubesUsed[cubeNum(row, col)][num]) {
            continue;
        }
        rowsUsed[row][num] = colsUsed[col][num] = cubesUsed[cubeNum(row, col)][num] = true;
        board[row][col] = (char) (num + '0');
        if (backtracking(row, col)) {
            return true;
        }
        board[row][col] = '.';
        rowsUsed[row][num] = colsUsed[col][num] = cubesUsed[cubeNum(row, col)][num] = false;
    }
    return false;
}

private int cubeNum(int i, int j) {
    int r = i / 3;
    int c = j / 3;
    return r * 3 + c;
}
```



#[51. N-Queens (Hard)](https://leetcode.com/problems/n-queens/description/)

[*n* 皇后问题研究的是如何将 *n* 个皇后放置在 *n*×*n* 的棋盘上，并且使皇后彼此之间不能相互攻击。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/8-queens.png)

上图为 8 皇后问题的一种解法。

给定一个整数 *n*，返回所有不同的 *n* 皇后问题的解决方案。

每一种解法包含一个明确的 *n* 皇后问题的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

**示例:**

```
输入: 4
输出: [
 [".Q..",  // 解法 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // 解法 2
  "Q...",
  "...Q",
  ".Q.."]
]
解释: 4 皇后问题存在两个不同的解法。
```

一行一行地摆放，在确定一行中的那个皇后应该摆在哪一列时，需要用三个标记数组来确定某一列是否合法，这三个标记数组分别为：列标记数组、45 度对角线标记数组和 135 度对角线标记数组。

45 度对角线标记数组的长度为 2 * n - 1，通过下图可以明确 (r, c) 的位置所在的数组下标为 r + c。

135 度对角线标记数组的长度也是 2 * n - 1，(r, c) 的位置所在的数组下标为 n - 1 - (r - c)。

```
private List<List<String>> solutions;
private char[][] nQueens;
private boolean[] colUsed;
private boolean[] diagonals45Used;
private boolean[] diagonals135Used;
private int n;

public List<List<String>> solveNQueens(int n) {
    solutions = new ArrayList<>();
    nQueens = new char[n][n];
    for (int i = 0; i < n; i++) {
        Arrays.fill(nQueens[i], '.');
    }
    colUsed = new boolean[n];
    diagonals45Used = new boolean[2 * n - 1];  //45度的对角线一共有2n-1条，
    diagonals135Used = new boolean[2 * n - 1];
    this.n = n;
    backtracking(0);
    return solutions;
}

private void backtracking(int row) {
    if (row == n) {
        List<String> list = new ArrayList<>();
        for (char[] chars : nQueens) {
            list.add(new String(chars));
        }
        solutions.add(list);
        return;
    }

    for (int col = 0; col < n; col++) {
        int diagonals45Idx = row + col; //计算出当前坐标是第几条45度对角线
        int diagonals135Idx = n - 1 - (row - col);
        if (colUsed[col] || diagonals45Used[diagonals45Idx] || diagonals135Used[diagonals135Idx]) {
            continue;
        }
        nQueens[row][col] = 'Q';
        colUsed[col] = diagonals45Used[diagonals45Idx] = diagonals135Used[diagonals135Idx] = true;
        backtracking(row + 1);
        colUsed[col] = diagonals45Used[diagonals45Idx] = diagonals135Used[diagonals135Idx] = false;
        nQueens[row][col] = '.';
    }
}
```