# TPC-H Q3

## Input Format

```
==> customer.txt <==
1|BUILDING
2|AUTOMOBILE
3|AUTOMOBILE
4|MACHINERY
5|HOUSEHOLD
6|AUTOMOBILE
7|AUTOMOBILE
8|BUILDING
9|FURNITURE
10|HOUSEHOLD

==> lineitem.txt <==
1|33203.72|1996-03-13
1|69788.52|1996-04-12
1|16381.28|1996-01-29
1|29767.92|1996-04-21
1|37596.96|1996-03-30
1|48267.84|1996-01-30
2|71798.72|1997-01-28
3|73200.15|1994-02-02
3|75776.05|1993-11-09
3|47713.86|1994-01-16

==> orders.txt <==
1|3689999|1996-01-02
2|7800163|1996-12-01
3|12331391|1993-10-14
4|13677602|1995-10-11
5|4448479|1994-07-30
6|5562202|1992-02-21
7|3913430|1996-01-10
32|13005694|1995-07-16
33|6695788|1993-10-27
34|6100004|1998-07-21
```

## Compile

```
mkdir -p build-dir-path
cd build-dir-path
cmake source-dir-path
make -j
```

## Run

```
./tpch-q3 -c /mnt/storage1/tpch-tables/customer.txt -o /mnt/storage1/tpch-tables/orders.txt -l /mnt/storage1/tpch-tables/lineitem.txt --cf test --of 1992-11-11 --lf 1992-11-11 --limit 5
```

