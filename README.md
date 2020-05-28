# LSTMで株価予測をしてみた

- LSTM自体はもう使い古された手法で、株価予測も多く実験されていますが、手の体操のつもりで、自分も単純なモデルを書いてみました。
- このモデルでは、50日前までの株や国債の記録を用いて、50日後の株と国債の動きを予測しようというものです。
- 結果は長ったらしい.pngファイルにあるとおり、ダメダメですが、良い練習にはなりました。

## 使用データ

- アメリカ国債の利率(USGB10)
- アメリカ株価Standard & Poor 500 (USSP500)
- 日経平均 (NIKKEI)
- 日本国債の利率 (JGBCM)
