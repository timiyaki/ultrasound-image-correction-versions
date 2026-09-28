# 版本特性

下表按各程序包名称、附带说明及代码功能整理。每个目录保留该版自己的 `使用说明.md`；后续版本不一定延续此前所有实验功能。特别是 v21 后把三维流程分离，v25–v28 的滤波链也经历了重新取舍。

| 版本 | 该版重点 | 原始程序包 |
|---|---|---|
| [v01](versions/v01/) | 早期二维 DSC 与灰度校正；α/β/γ 滑块、当前帧实时预览与自动调参。**编号为推定**。 | `UltrasoundBinCorrection_Auto_2D.zip` |
| [v02](versions/v02/) | 调整当前帧自动调色建议，增强亮暗分离。 | `UltrasoundBinCorrection_Auto_2D_v2.zip` |
| [v03](versions/v03/) | 整组图像逐帧自动估计参数。 | `UltrasoundBinCorrection_Group_Auto_v3.zip` |
| [v04](versions/v04/) | 加强自动模式的 γ 曲线。 | `UltrasoundBinCorrection_Group_Auto_v4_StrongGamma.zip` |
| [v05](versions/v05/) | 提高自动调色的对比度。 | `UltrasoundBinCorrection_Group_Auto_v5_HighContrast.zip` |
| [v06](versions/v06/) | 强化胎儿等主体区域的亮度、对比度和边界表现。 | `UltrasoundBinCorrection_Group_Auto_v6_SubjectEnhanced.zip` |
| [v07](versions/v07/) | 新增骨骼声影区域的可选灰度补偿。 | `UltrasoundBinCorrection_v7_BoneShadowCompensation.zip` |
| [v08](versions/v08/) | 增强边缘清晰度。 | `UltrasoundBinCorrection_v8_EdgeClarity.zip` |
| [v09](versions/v09/) | 调整导出对比图，突出 DSC 后调色前/后的比较。 | `UltrasoundBinCorrection_v9_DSC_Comparison.zip` |
| [v10](versions/v10/) | 自动调色链加入去噪、去散斑及边缘增强。 | `UltrasoundImageCorrection_Image_v10_DenoiseEdge.zip` |
| [v11](versions/v11/) | 减少真实轮廓旁的残影/重影。 | `UltrasoundImageCorrection_Image_v11_AntiGhost.zip` |
| [v12](versions/v12/) | 改为整组图像共用一套自动参数。 | `UltrasoundImageCorrection_Image_v12_SharedParameters.zip` |
| [v13](versions/v13/) | 亮区红色细线描边与连续区域扩散。 | `UltrasoundImageCorrection_Image_v13_RedOutline.zip` |
| [v14](versions/v14/) | 四宫格实时预览：原始、DSC、调色、描边；重新载入保留当前帧。 | `UltrasoundImageCorrection_Image_v14_FourPanelLive.zip` |
| [v15](versions/v15/) | 红线标整体亮区，蓝线标强反声区，并提供各自阈值。 | `UltrasoundImageCorrection_Image_v15_DualOutline.zip` |
| [v16](versions/v16/) | 主体自动识别、手动画笔标注及向连续帧传播。 | `UltrasoundImageCorrection_Image_v16_SubjectBrush.zip` |
| [v17](versions/v17/) | 根据扫描角度及红蓝标注初步重建可旋转三维图像。 | `UltrasoundImageCorrection_Image_v17_3D.zip` |
| [v18](versions/v18/) | 三维帧间插值、方向轴与二维截面联动。 | `UltrasoundImageCorrection_Image_v18_Continuous3D.zip` |
| [v19](versions/v19/) | 优化三维交互和切帧渲染性能。 | `UltrasoundImageCorrection_Image_v19_HighPerformance3D.zip` |
| [v20](versions/v20/) | 将切片式显示改进为插值后的连续体表面。 | `UltrasoundImageCorrection_Image_v20_SolidVoxelSurface.zip` |
| [v21](versions/v21/) | 回归独立二维程序；抑制轮廓旁残影，加强胎儿/子宫壁分离并避开边缘亮斑。 | `UltrasoundImageCorrection_Image_v21_FetalWallSeparation.zip` |
| [v22](versions/v22/) | 利用较亮反射线改善接触区域分界；三列导出原图、调色图、标注图。 | `UltrasoundImageCorrection_Image_v22_ReflectionBoundary.zip` |
| [v23](versions/v23/) | 增加独立的自动标注按钮；红/蓝阈值自动建议及手动滑块。 | `UltrasoundImageCorrection_Image_v23_AutoAnnotation.zip` |
| [v24](versions/v24/) | 加入可选 U-Net 胎儿 ROI 识别入口与模型放置说明。 | `UltrasoundImageCorrection_Image_v24_UNet_ROI.zip` |
| [v25](versions/v25/) | 二维论文式图组；加入可调 SVD 低秩去散斑、Hessian 结构滤波和 PFDTV 边缘保持模块，不含三维或机器学习。 | `UltrasoundImageCorrection_Image_v25_PaperPanels.zip` |
| [v26](versions/v26/) | 精简为自适应 Lee 去散斑与一次双边滤波，保留 DSC、α/β/γ、统一自动参数及红蓝描边。**原包内层仍误标 v25**。 | `UltrasoundImageCorrection_Image_v26_LeeBilateral.zip` |
| [v27](versions/v27/) | 在 v21 二维代码上局部优化：对数域自适应 Lee、单次 5×5 双边滤波、共同白点亮部限幅及可调强度；保留胎儿/子宫壁分离，不含三维或机器学习。 | [v27.zip](packages/v27.zip) |
| [v28](versions/v28/) | 从 v27 派生：采用 edgeEnhance-0913 的方向自适应边缘增强；滑块改名“边缘增强”；缓存组统计与中间图、单工作线程仅处理最新请求、快速/精确边缘预览；导出保持全分辨率。不含三维或机器学习。 | [v28.zip](packages/v28.zip) |

v01–v26 原始压缩包仅用于溯源，未放入仓库；v27/v28 提供无影像数据的源码压缩包。对比图和真实超声采集资料有意排除。
