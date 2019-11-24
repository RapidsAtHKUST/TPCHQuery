# TPCHQuery

## Index Files

Column store: `*.rapids`

## Create Symbolic Links of Files

```bash
./link_files.sh /data
```

## Compile 

```bash
./compile.sh
```

## Run

```bash
./run.sh customer.txt orders.txt lineitem.txt 1 BUILDING 1995-03-29 1995-03-27 5
```

* check with gt_testcase_4.txt

```bash
 ./run.sh customer.txt orders.txt lineitem.txt 3 BUILDING 1995-03-29 1995-03-27 5 BUILDING 1995-02-29 1995-04-27 10 BUILDING 1995-03-28 1995-04-27 2
```

* check with gt_testcase_3.txt

```bash
./run.sh customer.txt orders.txt lineitem.txt 1 BUILDING 1998-08-02 1992-01-02 5
```

## Source Files

see [q3-src](q3-src)

## Installed Object

In this folder with the name `tpch-q3-submit`. 