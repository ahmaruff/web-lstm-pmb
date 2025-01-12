# Sistem Prediksi Penerimaan Mahasiswa Baru
## Menggunakan model LSTM

> Sistem ini menggunakan data histori jumlah mahasiswa baru setiap tahunnya, untuk memprediksi jumlah mahasiswa 3 tahun berikutnya

## Cara Install

> Python dengan versi >= 3.10 wajib terinstall terlebih dahulu

1. Clone/salin project ini  
2. Buka terminal/CMD, dan pindah ke lokasi/folder project ini    
3. Buat virtual environment dengan perintah  
```
python -m venv .venv
```  


4. Aktifkan virtual environment yang telah dibuat dengan perintah  
```
.venv/Scripts\activate
```  

5. Install library yang dibutuhkan dengan perintah   
```
pip install -r requirements.txt
```

6. Inisiasi database (SQLITE) dengan perintah  
```
flask --app prediksipmb init-db
```  

7. Jalankan server dengan perintah  
```
flask --app prediksipmb run --debug
```    

8. buka website melalui browser (biasanya url-nya `http://127.0.0.1:5000` )  


> untuk mematikan server, cukup tekan `CTRL+C`

> untuk mematikan virtual environment, gunakan perintah `deactivate`  

### Cara menjalankan website (jika proses instalasi sudah dilakukan sebelumnya)
 
1. Buka terminal/CMD, dan pindah ke lokasi/folder project ini  
2. Aktifkan virtual environment yang telah dibuat dengan perintah  
```
.venv/Scripts\activate
```  

3. Jalankan server dengan perintah  
```
flask --app prediksipmb run --debug
```  

4. buka website melalui browser (biasanya url-nya `http://127.0.0.1:5000` )  
