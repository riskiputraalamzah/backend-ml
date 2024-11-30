const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const app = express();
const port = 3000;

// Setup Multer untuk menangani upload file
const storage = multer.memoryStorage(); // Menyimpan file dalam memori
const upload = multer({
  storage: storage,
  limits: { fileSize: 1000000 }, // Membatasi ukuran file menjadi 1MB
});

// Fungsi untuk memuat model TensorFlow
let model;

async function loadModel() {
  try {
    model = await tf.loadGraphModel("file://submissions-model/model.json");
    console.log("Model loaded");
  } catch (error) {
    console.error("Error loading model: ", error);
    throw new Error("Failed to load model.");
  }
}

loadModel(); // Memuat model saat aplikasi dimulai

// Endpoint untuk menerima gambar dan melakukan prediksi
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res
      .status(400)
      .json({ status: "fail", message: "No file uploaded" });
  }

  try {
    // Menangani error jika ukuran file lebih dari 1MB
    if (req.file.size > 1000000) {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }

    // Cek apakah file yang diupload adalah gambar
    if (!req.file.mimetype.startsWith("image/")) {
      return res
        .status(400)
        .json({ status: "fail", message: "Uploaded file is not an image" });
    }

    const imageBuffer = req.file.buffer;
    let imageTensor = tf.node.decodeImage(imageBuffer); // Decode image buffer

    // Resize image to 224x224 to match the model's input size
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);

    // Normalisasi gambar ke rentang [0, 1]
    imageTensor = imageTensor.div(255.0);

    // Cek shape tensor setelah resize dan normalisasi
    console.log(
      "Image Tensor shape after resize and normalization: ",
      imageTensor.shape
    );

    // Memastikan tensor memiliki dimensi yang benar
    if (imageTensor.shape.length === 3) {
      imageTensor = imageTensor.expandDims(0); // Tambahkan batch dimension
    }

    // Lakukan prediksi menggunakan model
    const prediction = model.predict(imageTensor);

    // Ambil hasil prediksi berupa array dengan rentang nilai [0, 1]
    const predictionData = prediction.dataSync();
    const predictedValue = parseFloat(predictionData[0].toFixed(3));

    console.log("Prediction Data: ", predictionData[0]); // Debugging hasil prediksi
    console.log("Prediction Data: ", predictedValue); // Debugging hasil prediksi

    // Prediksi pertama (misalnya untuk "Cancer")
    const result = predictedValue > 0.58 ? "Cancer" : "Non-cancer";

    // Tentukan saran berdasarkan hasil prediksi
    const suggestion =
      result === "Cancer"
        ? "Segera periksa ke dokter!"
        : "Penyakit kanker tidak terdeteksi.";

    // Simpan hasil prediksi ke file lokal (JSON)
    const resultData = {
      id: generateUUID(),
      result: result,
      suggestion: suggestion,
      createdAt: new Date().toISOString(),
    };

    // Simpan ke file JSON lokal
    const filePath = path.join(__dirname, "predictions.json");
    const predictions = fs.existsSync(filePath)
      ? JSON.parse(fs.readFileSync(filePath))
      : [];
    predictions.push(resultData);
    fs.writeFileSync(filePath, JSON.stringify(predictions, null, 2));

    // Kembalikan response ke pengguna
    res.json({
      status: "success",
      message: "Model is predicted successfully",
      data: resultData,
    });
  } catch (error) {
    console.error("Error during prediction: ", error);
    res.status(400).json({
      status: "fail",
      message: `Terjadi kesalahan dalam melakukan prediksi`,
    });
  }
});

// Endpoint untuk mendapatkan riwayat prediksi
app.get("/predict/histories", async (req, res) => {
  try {
    const filePath = path.join(__dirname, "predictions.json");

    if (fs.existsSync(filePath)) {
      const predictions = JSON.parse(fs.readFileSync(filePath));

      // Format data sesuai dengan ketentuan
      const formattedPredictions = predictions.map((prediction) => ({
        id: prediction.id,
        history: {
          result: prediction.result,
          createdAt: prediction.createdAt,
          suggestion: prediction.suggestion,
          id: prediction.id,
        },
      }));

      res.json({
        status: "success",
        data: formattedPredictions,
      });
    } else {
      res.json({
        status: "success",
        data: [],
      });
    }
  } catch (error) {
    console.error("Error fetching histories:", error);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan saat mengambil riwayat prediksi",
    });
  }
});

// Fungsi untuk menghasilkan UUID (ID unik untuk setiap prediksi)
function generateUUID() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Menangani error file terlalu besar (lebih dari 1MB)
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  next(err); // Lanjutkan ke error handler lainnya
});

// Jalankan server
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
