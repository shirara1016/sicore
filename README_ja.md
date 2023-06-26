# sicoreパッケージ
Selective Inferenceで共通して利用される機能をまとめたパッケージです．

## 導入
インストールにはPython3.10以上が必要です．また依存パッケージは自動的にインストールされます．もしtensorflowやpytorchのtensorを利用したい場合は手動で対応するフレームワークをインストールしてください．
```
$ pip install sicore
```
アンインストール：
```
$ pip uninstall sicore
```

## APIドキュメント
詳細なAPIドキュメントは[こちら](https://shirara1016.github.io/sicore/)です．

## 機能一覧
`from sicore import *`でインポートされる機能は以下の通りです．

**検定**
- NaiveInferenceNorm：正規分布に従う検定統計量に対するnaiveな検定
- SelectiveInferenceNorm：正規分布に従う検定統計量に対するselectiveな検定
    - Parametric SIとOver-Coniditinonigに対応
    - Parametric SIでは次の3種類の方法が利用可能
        - 精度保証付きのp値の算出
        - 帰無仮説が棄却されるかどうかの判定
        - 指定した全範囲のparametric探索
- NaiveInferenceChi：カイ分布に従う検定統計量に対するnaiveな検定
- SelectiveInferenceChi：カイ分布に従う検定統計量に対するselectiveな検定
    - Parametric SIとOver-Coniditinonigに対応
    - Parametric SIでは次の3種類の方法が利用可能
        - 精度保証付きのp値の算出
        - 帰無仮説が棄却されるかどうかの判定
        - 指定した全範囲のparametric探索

**切断分布**
全て複数の切断区間に対応し，mpmathを用いた任意精度の計算
- tn_cdf()：切断正規分布
- tc_cdf()：切断カイ分布

**検定の評価**
- type1_error_rate()
- power()

**図の描画**
- pvalues_hist()：p値のヒストグラムを描画
- pvalues_qqplot()：p値の一様Q-Qプロットを描画

**区間**
- intervals.intersection()：2つの区間の積を計算
- intervals.intersection_all()：複数区間の積を計算
- intervals.union_all()：複数区間の和を計算
- intervals.not_()：実数上で区間の補集合を計算

**その他の便利な機能**
- OneVec：特定の場所が1，それ以外が0のベクトルを生成
- poly_lt_zero()：多項式の0以下となる区間を計算
- polytope_to_interval()：二次形式の選択イベントを切断区間へと変換する関数
- construct_projection_matrix()：基底からそれらが張る部分空間への射影行列を生成

## その他
テストの実行：
```
$ pytest tests/
```
