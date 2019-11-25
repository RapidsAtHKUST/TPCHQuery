# TPCHQuery - Rapids团队

## 主要设计思想

### 索引文件

#### Customer join Order 索引

首先把`Customer`和`Order`表join成一张表, 利用 `order_date` 和 `customer_category` 来分`partition`， 
把相同`(customer_category, order_date)`的行group到同一个partition。
我们实现了基于OpenMP的BucketSort实现这一group(对行做reorder)的目的, 
并对每个partition按照`order-key`进行排序来提高之后join时候的内存访问locality。 
为了方便join LineItem表, 我们把reorder之后的行存储转为了列存储, 
存到了 `orders.txt_ORDER_KEY.rapids`, `orders.txt_ORDER_DATE.rapids` 以及meta数据 `orders.txt_ORDER_META.rapids`。
其中meta数据包含了: `partition`信息, `category`映射信息, 还有`min-max order_date`信息。

#### LineItem 索引

类似之前的partition思路, 我们按照`ship_date`进行`partition`， 并且将reorder之后的行存储转为了列存储, 
存到了`lineitem.txt_LINE_ITEM_KEY.rapids`, `lineitem.txt_LINE_ITEM_PRICE.rapids` (之后aggregation用) 以及meta数据 `lineitem.txt_LINE_ITEM_META.rapids`。
其中meta数据包含了: `max order id`,  `partition`信息, `min-max ship_date` 信息。

### Join算法

Q3中有三个filter, Customer和Order有关的两个filter定位到对应的Order表中`partition`； LineItem有关的filter定位到对应的LineItem表中`partitions`。
然后对Order表`filtered partions`建立bitmap-index, 用LineItem表进行probe和aggregation， 最终选出非0的行按照`price`选出top-k。

### GPU Join Acceleration

GPU负责对`keys`进行`intersection`, 之后的`price`有关的aggregation由CPU来做。 
多GPU任务通过分割扫LineItem的`filtered partition`来实现。

## 详细测试步骤

以下测试需要在当前目录进行。

### 链接指定路经下的三个input文件 (`/data`下的三个文件: customer.txt orders.txt lineitem.txt)

```bash
./link_files.sh /data
```

### 编译 (默认启用GPU)

```bash
./compile.sh
```

### 运行 (默认使用两个GPU)

```bash
./run.sh customer.txt orders.txt lineitem.txt 1 BUILDING 1995-03-29 1995-03-27 5
```

* check with `gt_testcase_4.txt`

```bash
 ./run.sh customer.txt orders.txt lineitem.txt 3 BUILDING 1995-03-29 1995-03-27 5 BUILDING 1995-02-29 1995-04-27 10 BUILDING 1995-03-28 1995-04-27 2
```

* check with `gt_testcase_3.txt`

```bash
./run.sh customer.txt orders.txt lineitem.txt 1 BUILDING 1998-08-02 1992-01-02 5
```

## 代码目录

请见 [q3-src](q3-src)。

## 编译和运行方法

在详细测试步骤中已经提到。

## 安装的可执行文件

在当前工作目录，名字为`tpch-q3-submit`。