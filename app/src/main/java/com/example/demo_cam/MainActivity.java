package com.example.demo_cam;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.SeekBar;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import de.hdodenhof.circleimageview.CircleImageView;
import jp.co.cyberagent.android.gpuimage.GPUImageView;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2BGR;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class MainActivity extends AppCompatActivity {
    private String cameraId;
    private AutoFitTextureView textureView;
    private GPUImageView gpuImageView;
    private CameraDevice cameraDevice;
    private Size previewSize;
    private CaptureRequest previewCaptureRequest;
    private CaptureRequest.Builder previewCaptureRequestBuilder;
    private CameraCaptureSession cameraCaptureSession;
    private Button btnRatio;
    private ImageButton btnFlash;
    private SeekBar seekBar;
    private CircleImageView image;
    private ImageButton btnCapture;
    private ImageButton btnChangeCamera;
    private int ratioNumber = 34;
    private boolean flash = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textureView = (AutoFitTextureView)findViewById(R.id.textureView);
        gpuImageView = (GPUImageView)findViewById(R.id.gpuimageview);
        btnCapture = findViewById(R.id.btnCapture);
        btnChangeCamera = findViewById(R.id.btnChangeCamera);
        btnFlash = findViewById(R.id.btnFlash);
        btnRatio = findViewById(R.id.btnRatio);
        seekBar = findViewById(R.id.seekBar);
        image = findViewById(R.id.image);

        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");
        fg = new Mat();
        mskmat = new Mat();
        invmskmat = new Mat();
        resmat = new Mat();

        btnRatio.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch (ratioNumber){
                    case 34:
                        btnRatio.setText("1 : 1");
                        ratioNumber = 11;
                        break;
                    case 11:
                        btnRatio.setText("4 : 5");
                        ratioNumber = 45;
                        break;
                    case 45:
                        btnRatio.setText("9 : 16");
                        ratioNumber = 916;
                        break;
                    case 916:
                        btnRatio.setText("3 : 4");
                        ratioNumber = 34;
                        break;
                    default:
                        break;
                }
            }
        });

        btnFlash.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(flash){
                    btnFlash.setImageResource(R.mipmap.none_flash);
                    flash = false;
                }
                else {
                    btnFlash.setImageResource(R.mipmap.flash);
                    flash = true;
                }
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            setupCamera(textureView.getWidth(), textureView.getHeight());
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(surfaceTextureListener);
        }
    }

    @Override
    protected void onPause() {
        stopBackgroundThread();
        closeCamera();
        super.onPause();
    }

    private TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            setupCamera(width, height);
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void setupCamera(int width, int height) {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String id : cameraManager.getCameraIdList()) {
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(id);

                if (cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }
                StreamConfigurationMap map =
                        cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                // Set Size để hiển thị lên màn hình
                previewSize = getPreferredPreviewsSize(
                        map.getOutputSizes(SurfaceTexture.class),
                        width,
                        height);
                cameraId = id;
                break;
            }
            textureView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private Size getPreferredPreviewsSize(Size[] mapSize, int width, int height) {
        List<Size> collectorSize = new ArrayList<>();
        for (Size option : mapSize) {
            if (width > height) {
                if (option.getWidth() > width && option.getHeight() > height) {
                    collectorSize.add(option);
                }
            } else {
                if (option.getWidth() > height && option.getHeight() > width) {
                    collectorSize.add(option);
                }
            }
        }
        if (collectorSize.size() > 0) {
            return Collections.min(collectorSize, new Comparator<Size>() {
                @Override
                public int compare(Size lhs, Size rhs) {
                    return Long.signum(lhs.getWidth() * lhs.getHeight() - rhs.getHeight() * rhs.getWidth());
                }
            });
        }
        return mapSize[0];
    }

    private void openCamera() {
        CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 100);
                return;
            }
            cameraManager.openCamera(cameraId, cameraDeviceStateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera(){
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    private CameraDevice.StateCallback cameraDeviceStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            camera.close();
            cameraDevice = null;
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            camera.close();
            cameraDevice = null;
        }
    };

    private CameraCaptureSession.CaptureCallback cameraSessionCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureStarted(CameraCaptureSession session, CaptureRequest request,
                                             long timestamp, long frameNumber) {
                    super.onCaptureStarted(session, request, timestamp, frameNumber);
                }

                @Override
                public void onCaptureCompleted(CameraCaptureSession session,
                                               CaptureRequest request, TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                }

                @Override
                public void onCaptureFailed(CameraCaptureSession session,
                                            CaptureRequest request, CaptureFailure failure) {
                    super.onCaptureFailed(session, request, failure);
                }
            };

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);

            previewCaptureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            previewCaptureRequestBuilder.addTarget(previewSurface);

            init();
            startBackgroundThread();
            cameraDevice.createCaptureSession(Arrays.asList(previewSurface),
                    // Hàm Callback trả về kết quả khi khởi tạo.
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            if (cameraDevice == null) {
                                return;
                            }
                            try {
                                previewCaptureRequest = previewCaptureRequestBuilder.build();
                                cameraCaptureSession = session;
                                cameraCaptureSession.setRepeatingRequest(
                                        previewCaptureRequest,
                                        cameraSessionCaptureCallback,
                                        backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            Toast.makeText(getApplicationContext(),
                                    "Create camera session fail", Toast.LENGTH_SHORT).show();
                        }
                    },
                    null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Object lock = new Object();
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private boolean runsegmentor = false;

    private void startBackgroundThread(){
        backgroundThread = new HandlerThread("haizzz");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
        synchronized (lock) {
            runsegmentor = true;
        }
        backgroundHandler.post(periodicSegment);
    }

    private void stopBackgroundThread() {
        try {
            backgroundThread.quitSafely();
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
            synchronized (lock) {
                runsegmentor = false;
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "Interrupted when stopping background thread", e);
        }catch (Exception e){

        }
    }

    private Runnable periodicSegment = new Runnable() {
        @Override
        public void run() {
            synchronized (lock) {
                if (runsegmentor) {
                    segment_frame();
                }
            }
            try {
                backgroundHandler.post(periodicSegment);
            }catch (Exception e){

            }
        }
    };
    private void segment_frame() {
        if (cameraDevice == null) {
            return;
        }
        long t1  = SystemClock.uptimeMillis();
        fgd = textureView.getBitmap(frame_width, frame_height);
        bitmap = fgd.createScaledBitmap(fgd, W, H, true);
        long t2 = SystemClock.uptimeMillis();
        Log.e(TAG, "t get bit map: "+(t2-t1));
        t1 = SystemClock.uptimeMillis();
        segment();
        t2 = SystemClock.uptimeMillis();
        Log.e(TAG, "t total: "+(t2-t1) + " FPS: " + (1000.0/(t2-t1)));
//        bitmap.recycle();

        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                if(result!=null) {
                    gpuImageView.setImage(result);
                }
            }
        });
    }

    String TAG = "______________________________";
    int W = 128, H = 128, frame_height = 624, frame_width = 416;
    Interpreter.Options tfliteOptions = new Interpreter.Options();
    MappedByteBuffer tfliteModel;
    Interpreter tflite;
    int[] intValues = new int[W*H];
    ByteBuffer imgData = ByteBuffer.allocateDirect(W * H * 12);
    Bitmap result = Bitmap.createBitmap(frame_width, frame_height, Bitmap.Config.ARGB_8888);
    GpuDelegate gpuDelegate = new GpuDelegate();
    float[][][][] segmap = new float[1][W][H][2];
    Mat invmskmat, resmat, fg, mskmat;
    float[] tmp = new float[W*H];
    Bitmap resbmp = Bitmap.createBitmap(frame_width, frame_height, Bitmap.Config.ARGB_8888);
    private Bitmap fgd, bitmap;

    void init() throws IOException {
        tfliteModel = loadModelFile(this);
        tfliteOptions.addDelegate(gpuDelegate);
        tfliteOptions.setNumThreads(4);
        tfliteOptions.setAllowFp16PrecisionForFp32(true);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        imgData.order(ByteOrder.nativeOrder());
    }

    void segment() {
        if(tflite==null) {
            result = fgd;
            return;
        }
        long t1 = SystemClock.uptimeMillis();
        convertBitmapToByteBuffer(bitmap);
        long t2 = SystemClock.uptimeMillis();
        Log.e(TAG, "t convert bitmap to input: "+(t2-t1));

        tflite.run(imgData, segmap);
        long t3 = SystemClock.uptimeMillis();
        Log.e(TAG, "t run model: "+(t3-t2));

        //        inputToBitmap(fgd);
        imageblend(fgd);
        long t4 = SystemClock.uptimeMillis();
        Log.e(TAG, "t convet output to bit map" + (t4-t3));
    }

    void convertBitmapToByteBuffer(Bitmap bitmap) {
        imgData.rewind();
        bitmap.getPixels(intValues, 0, W, 0, 0, W, H);
        int pixel = 0;
        for (int i = 0; i < W; ++i) {
            for (int j = 0; j < H; ++j) {
                int val = intValues[pixel++];
                imgData.putFloat(((val >> 16) & 0xFF));
                imgData.putFloat(((val >> 8) & 0xFF));
                imgData.putFloat((val & 0xFF));
            }
        }
    }

    void imageblend(Bitmap fg_bmp){
        if (segmap!=null){
            int sz = 0;
//      long startTime = SystemClock.uptimeMillis();
            for(int i = 0; i < W; i++) for(int j = 0; j < H; j++) {
                if(segmap[0][i][j][1]>=0.5)
                    tmp[sz] = 1;
                else
                    tmp[sz] = 0;
                sz++;
            }
//      long endTime = SystemClock.uptimeMillis();
//      Log.e(TAG, "t convert 2d -> 1d: "+(endTime-startTime));
            mskmat = new Mat(H, W, CV_32F);
            mskmat.put(0,0, tmp);
            Imgproc.medianBlur(mskmat, mskmat, 5);

            Imgproc.cvtColor(mskmat, mskmat, Imgproc.COLOR_GRAY2BGR);
            invmskmat = new Mat(frame_height, frame_width, CV_32FC3, new Scalar(1.0, 1.0, 1.0));
            Imgproc.resize(mskmat, mskmat, new org.opencv.core.Size(frame_width, frame_height));
            Core.subtract(invmskmat, mskmat, invmskmat);

            Utils.bitmapToMat(fg_bmp, fg);
            Imgproc.cvtColor(fg, fg, COLOR_BGRA2BGR);
            fg.convertTo(fg, CV_32FC3, 1.0/255.0);

            Core.multiply(mskmat, fg, resmat);
            Imgproc.GaussianBlur(fg, fg, new org.opencv.core.Size(9,9),0);
            Core.multiply(fg, invmskmat, fg);
            Core.add(fg, resmat, resmat);

            resmat.convertTo(resmat, CV_8UC3,255.0);
            Utils.matToBitmap(resmat, resbmp);
            result = resbmp;
        }else{
            result = fg_bmp;
        }
    }

    MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("deeplab_v2_128_test.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    void close() {
        if(tflite!=null){
            tflite.close();
            tflite = null;
        }
        gpuDelegate.close();
        gpuDelegate = null;
        tfliteModel = null;
    }
}