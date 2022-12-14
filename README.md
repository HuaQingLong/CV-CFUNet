# CV-CFUNet
The implementation of CV-CFUNet using tensorflow. (CV-CFUNet: Complex-Valued Channel Fusion UNet for Refocusing of Ship Targets in SAR Images)

************************************************************
The full code will be available when the paper is published.
************************************************************

Abstract:
In a synthetic aperture radar (SAR) system, target rotation during the coherent integration time results in a time-varying Doppler frequency shift and a blurred image. Blurred images are not conducive to subsequent information interpretation. This paper proposes a complex-valued channel fusion U-shape network (CV-CFUNet) for the three-dimensional rotation refocusing task of ship targets. The proposed method integrates the refocusing task into a blind inverse problem. To take advantage of the amplitude and phase information of complex SAR images, all elements of CV-CFUNet including convolutional layer, activation function, feature maps, and network parameters are extended to the complex domain. The proposed CV-CFUNet is designed by adopting a complex-valued encoder (CV-Encoder), channel fusion module (CFM), and complex-valued decoder (CV-Decoder) to adaptively learn complex features. Experiments on simulated data, GF-3 data, and Sentinel-1 data show that the proposed method achieves significant improvements over existing methods in both efficiency and refocusing accuracy.
