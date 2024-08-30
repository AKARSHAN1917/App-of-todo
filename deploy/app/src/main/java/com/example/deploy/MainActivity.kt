package com.example.deploy

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import com.example.deploy.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var selectBtn: Button
    lateinit var predictBtn: Button
    lateinit var resultView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // Initialize labels
        labels = application.assets.open("labels.txt").bufferedReader().readLines()

        selectBtn = findViewById(R.id.selectBtn)
        predictBtn = findViewById(R.id.predictBtn)
        resultView = findViewById(R.id.resultView)
        imageView = findViewById(R.id.imageView)

        // Image processor
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        selectBtn.setOnClickListener {
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, 100)
        }

        predictBtn.setOnClickListener {
            if (::bitmap.isInitialized) {
                try {
                    var tensorImage = TensorImage(DataType.UINT8)
                    tensorImage.load(bitmap)

                    tensorImage = imageProcessor.process(tensorImage)
                    val model = MobilenetV110224Quant.newInstance(this)

                    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                    inputFeature0.loadBuffer(tensorImage.buffer)

                    val outputs = model.process(inputFeature0)
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                    var maxIdx = 0
                    outputFeature0.forEachIndexed { index, fl ->
                        if (outputFeature0[maxIdx] < fl) {
                            maxIdx = index
                        }
                    }

                    resultView.text = labels[maxIdx]

                    model.close()
                } catch (e: Exception) {
                    e.printStackTrace() // Handle exception during processing
                }
            } else {
                // Handle the case where bitmap is not initialized
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 100 && resultCode == RESULT_OK) {
            val uri = data?.data
            if (uri != null) {
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                    imageView.setImageBitmap(bitmap)
                } catch (e: Exception) {
                    e.printStackTrace() // Handle exception
                }
            }
        } else if (resultCode == RESULT_CANCELED) {
            // Handle the case where the user canceled the operation

        }
    }
}