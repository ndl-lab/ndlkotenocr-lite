# NDL古典籍OCR-Liteのデスクトップアプリケーション

NDL古典籍OCR-Liteの使い方を説明します。

## 起動方法
お使いのOSに対応する圧縮ファイルを展開し、ndlkotenocr_lite_guiをダブルクリック等で実行してください。

なお、セキュリティに関する警告が出ることがあります。

Windows 10の場合は、「WindowsによってPCが保護されました」→「詳細情報」→「実行」を選んでください。

macOS（Sequoia）の場合は、https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac
の手順に従ってください。

なお、初回の起動には1分程度時間を要することがあります。お待ちください。

## 操作方法



## ビルド方法
本アプリケーションは[Flet（外部サイト）](https://flet.dev/)を利用します。

いずれのOSの場合にも事前にFlutter-SDKの導入が必要です。依存関係のインストールに関する説明は省略します。

### Windowsの場合
https://flet.dev/docs/publish/windows/
も参照してください。
```
#(コマンドプロンプトを利用、ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
.\ocrenv\Scripts\activate
pip install flet==0.24.1
xcopy ..\src .\src
flet build windows  --template flet-build-template
```

### Macの場合

```
#(ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build macos  --template flet-build-template
```

### Linuxの場合
```
#(ndlkotenocr-lite-guiと同階層で実行する)
python3 -m venv ocrenv
source ./ocrenv/bin/activate
pip install flet==0.24.1
cp -r ../src .
flet build linux --template flet-build-template
```