%md
# クラスターの作り方  
## Driver を CPU / Worker を GPU にして GPU のムダを減らす（Databricks CLI）

Spark のマルチノード GPU クラスタでは、Driver 側の GPU が十分に使われず「遊ぶ」ことがあります。  
この Tips は **Worker は GPU**, **Driver は CPU** にして、GPU リソースのムダを減らす構成のクラスタを **CLI から作る**手順です。

---

## 0. 事前準備（ここだけ最初に確認）
- Databricks ワークスペースにアクセスできる
- クラスタ作成権限がある
- ローカル端末でコマンドを実行できる
- **Databricks CLI（最新版）** が使える

### CLI バージョン確認
```bash
databricks -v
```

---

## 1. CLI でログイン（OAuth が一番簡単）
> 重要：**workspace の URL**（例：`https://dbc-xxxx.cloud.databricks.com`）でログインします。  
> ※ account console の URL でログインすると、workspace 操作で失敗することがあります。

```bash
databricks auth login --host https://<YOUR-WORKSPACE-HOST>
```

---

## 2. 利用可能な Spark Runtime（spark_version）を選ぶ
```bash
databricks clusters spark-versions | grep gpu
```

出てきた一覧から、GPU 付きの Runtime を 1 つ選びます。  
以降、この値を **`<SPARK_VERSION>`** として使います（例：`17.3.x-gpu-ml-scala2.13`）。

---

## 3. 利用可能なノードタイプ（インスタンスタイプ）を選ぶ
```bash
databricks clusters list-node-types
```

ここで次を決めます：

- Worker 用（GPU）：`node_type_id`
- Driver 用（CPU）：`driver_node_type_id`

---

## 4. クラスタ定義 JSON を作る（コピペ用テンプレ）
### 4.1 JSON ファイルを作成
`cluster-hetero.json` という名前でファイルを作り、以下を貼り付けます。

> 注意：以下は **Azure の例**です。AWS / GCP の場合は項目名や node_type_id が異なります。  
> まずは `node_type_id` と `driver_node_type_id` を **あなたの環境の値に置き換える**のが最優先です。

```json
{
  "cluster_name": "hiroshi-gpu-workers-cpu-driver-demo",
  "spark_version": "17.3.x-gpu-ml-scala2.13",
  "node_type_id": "Standard_NC24ads_A100_v4",
  "driver_node_type_id": "Standard_D4ds_v5",
  "autoscale": { "min_workers": 4, "max_workers": 4 },
  "azure_attributes": {
    "availability": "ON_DEMAND_AZURE"
  }
}
```

---

## 5. クラスタを作成する（CLI）
```bash
databricks clusters create --json @cluster-hetero.json
```

例：
```bash
databricks clusters create --json @cluster-hetero.json
```

成功すると `cluster_id` が返るので控えます。

---

## 6. 起動状態を確認する（RUNNING まで）
```bash
databricks clusters get <CLUSTER_ID>
```

`PENDING` → `RUNNING` になれば OK です。

---

## 7. 設定が狙い通りか確認する
`databricks clusters get` の出力（または UI）で、以下を確認します：

- `node_type_id` が GPU（Worker）
- `driver_node_type_id` が CPU（Driver）

---

## 8. 後片付け（不要なら削除）
```bash
databricks clusters delete <CLUSTER_ID>
```
