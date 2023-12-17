package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/ibd1279/vks"
)

func init() {
	runtime.LockOSThread()
}

const (
	WindowWidth  = 800
	WindowHeight = 600
)

// Option type, for tracking some parts of pipeline creation
type Option[T any] struct {
	v   T
	set bool
}

func Some[T any](value T) Option[T] {
	return Option[T]{v: value, set: true}
}
func None[T any]() Option[T] {
	return Option[T]{set: false}
}
func (option Option[T]) IsSet() bool { return option.set }
func (option Option[T]) Some() T {
	return option.SomeOr(func() T { panic("attempt to get from None") })
}
func (option Option[T]) SomeOr(callback func() T) T {
	if option.set {
		return option.v
	}
	return callback()
}

// 32-bit Words
type WordsUint32 []uint32

func NewWordsUint32(b []byte) WordsUint32 {
	r := bytes.NewReader(b)
	words := make([]uint32, len(b)/4)
	binary.Read(r, binary.LittleEndian, words)
	return WordsUint32(words)
}

func (words WordsUint32) Sizeof() uint64 {
	return uint64(len(words) * 4)
}

// Main function.
func main() {
	vks.Init().OrPanic()
	defer vks.Destroy()

	var version uint32
	if result := vks.EnumerateInstanceVersion(&version); result.IsSuccess() {
		log.Printf("%v - API version", vks.ApiVersion(version))
		log.Printf("%v - vk.xml version", vks.VK_HEADER_VERSION_COMPLETE)
	}

	app := TriangleApplication{
		SelectInstanceLayers: []string{
			"VK_LAYER_KHRONOS_validation",
		},
		SelectInstanceExtensions: []string{
			vks.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
			vks.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
			vks.VK_KHR_SURFACE_EXTENSION_NAME,
			vks.VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
		},
		SelectDeviceExtensions: []string{
			vks.VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
			vks.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		},
		FramesInFlight: 2,
	}
	app.Run()
}

type TriangleApplication struct {
	window   *glfw.Window
	instance vks.InstanceFacade

	SelectInstanceLayers     []string
	SelectInstanceExtensions []string
	SelectDeviceExtensions   []string
	FramesInFlight           uint

	surface           vks.SurfaceKHR
	physicalDevice    vks.PhysicalDeviceFacade
	graphicQueueIndex uint32
	presentQueueIndex uint32
	graphicQueue      vks.QueueFacade
	presentQueue      vks.QueueFacade
	device            vks.DeviceFacade

	swapchain             vks.SwapchainKHR
	swapchainImgs         []vks.Image
	swapchainImgFmt       vks.Format
	swapchainExtent       vks.Extent2D
	swapchainImgViews     []vks.ImageView
	renderPass            vks.RenderPass
	swapchainFramebuffers []vks.Framebuffer
	pipelineLayout        vks.PipelineLayout
	pipelines             []vks.Pipeline

	graphicCommandPool    vks.CommandPoolFacade
	graphicCommandBuffers []vks.CommandBuffer

	imageAvailableSemaphores []vks.Semaphore
	renderFinishedSemaphores []vks.Semaphore
	inFlightFences           []vks.Fence
	imagesInFlight           []vks.Fence
	currentFrame             uint

	framebufferResize bool
}

func (app *TriangleApplication) glfwSetup() error {
	// Initialize GLFW
	err := glfw.Init()
	if err != nil {
		return err
	}

	// Tell GLFW we aren't using OpenGL.
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)

	// Create the window object.
	app.window, err = glfw.CreateWindow(WindowWidth, WindowHeight, "vks tutorial-triangle", nil, nil)
	if err != nil {
		return err
	}

	app.window.SetFramebufferSizeCallback(func(*glfw.Window, int, int) {
		app.framebufferResize = true
	})
	return nil
}
func (app *TriangleApplication) vulkanSetup() error {
	var extensions []string

	// This is a debug function, mostly to dump the instance options.
	// Useful to understand Whats available and what isn't working.
	enumerateInstanceOptions := func() error {
		var count uint32
		result := vks.EnumerateInstanceLayerProperties(&count, nil)
		if result.IsError() {
			return result.AsErr()
		}
		layerProperties := make([]vks.LayerProperties, count)
		result = vks.EnumerateInstanceLayerProperties(&count, layerProperties)
		if result.IsError() {
			return result.AsErr()
		}

		for _, layer := range append([]string{""}, app.SelectInstanceLayers...) {
			ln := vks.NewCString(layer)
			defer vks.FreeCString(ln)

			result = vks.EnumerateInstanceExtensionProperties(
				ln,
				&count,
				nil,
			)
			if result.IsError() {
				return result.AsErr()
			}
			extensionProperties := make([]vks.ExtensionProperties, count)
			result = vks.EnumerateInstanceExtensionProperties(
				ln,
				&count,
				extensionProperties,
			)
			if result.IsError() {
				return result.AsErr()
			}
			for h, ext := range extensionProperties {
				log.Printf("%s Ext%2d:%s / %v", layer, h,
					vks.ToString(ext.ExtensionName()),
					vks.ApiVersion(ext.SpecVersion()))
			}
		}
		return nil
	}

	if err := enumerateInstanceOptions(); err != nil {
		return err
	}

	createInstance := func() error {
		// See
		// https://github.com/ibd1279/vulkangotutorial/blob/main/tutorial/part03.md#a-common-pattern
		// for an explanation of the common pattern

		// Create the info object
		extensions = append(
			app.SelectInstanceExtensions,
			app.window.GetRequiredInstanceExtensions()...,
		)
		appInfo := vks.ApplicationInfo{}.
			WithDefaultSType().
			WithApplication("tutorial-triangle", vks.MakeApiVersion(0, 0, 1, 0)).
			WithEngine("NoEngine", vks.MakeApiVersion(0, 1, 0, 0)).
			WithApiVersion(uint32(vks.VK_API_VERSION_1_3)).
			AsCPtr()
		defer appInfo.Free()
		createInfo := vks.InstanceCreateInfo{}.
			WithDefaultSType().
			WithPApplicationInfo(appInfo).
			WithFlags(vks.InstanceCreateFlags(vks.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR)).
			WithLayers(app.SelectInstanceLayers).
			WithExtensions(extensions).
			AsCPtr()
		defer createInfo.Free()

		// create the result object
		var vkInstance vks.Instance

		// call the vulkan function
		if result := vks.CreateInstance(createInfo, nil, &vkInstance); result.IsError() {
			enumerateInstanceOptions()
			return result.AsErr()
		}

		// update the application
		app.instance = vks.MakeInstanceFacade(vkInstance)
		return nil
	}

	if err := createInstance(); err != nil {
		return err
	}

	// This is a debug function, mostly to communicate what devices were
	// available. The checked in code below just blindly chooses the first
	// option, because this is a tutorial.
	enumeratePhysicalDevices := func() error {
		var count uint32
		result := app.instance.EnumeratePhysicalDevices(&count, nil)
		if result.IsError() {
			return result.AsErr()
		}
		physicalDevices := make([]vks.PhysicalDevice, count)
		result = app.instance.EnumeratePhysicalDevices(&count, physicalDevices)
		if result.IsError() {
			return result.AsErr()
		}

		for k, phyDev := range physicalDevices {
			// This is how you chain the different structures
			// together for calls that depend on PNext
			// TODO see if there is a better way to chain the pnext
			// calls and hide the unsafe.
			phyDev := app.instance.MakePhysicalDeviceFacade(phyDev)
			driverProps := vks.PhysicalDeviceDriverProperties{}.
				WithDefaultSType().
				AsCPtr()
			defer driverProps.Free()
			vulkan11Props := vks.PhysicalDeviceVulkan11Properties{}.
				WithDefaultSType().
				WithPNext(unsafe.Pointer(driverProps)).
				AsCPtr()
			defer vulkan11Props.Free()
			vulkan12Props := vks.PhysicalDeviceVulkan12Properties{}.
				WithDefaultSType().
				WithPNext(unsafe.Pointer(vulkan11Props)).
				AsCPtr()
			defer vulkan12Props.Free()
			vulkan13Props := vks.PhysicalDeviceVulkan13Properties{}.
				WithDefaultSType().
				WithPNext(unsafe.Pointer(vulkan12Props)).
				AsCPtr()
			defer vulkan13Props.Free()
			props := vks.PhysicalDeviceProperties2{}.
				WithDefaultSType().
				WithPNext(unsafe.Pointer(vulkan13Props)).
				AsCPtr()
			defer props.Free()

			phyDev.GetPhysicalDeviceProperties2(props)

			name := vks.ToString(props.Properties().DeviceName())
			apiVersion := vks.ApiVersion(props.Properties().ApiVersion())
			devType := props.Properties().DeviceType()
			driverVersion := vks.ApiVersion(props.Properties().DriverVersion())
			vendorId := props.Properties().VendorID()
			driverInfo := vks.ToString(driverProps.DriverInfo())
			driverName := vks.ToString(driverProps.DriverName())
			pointClippingBehavior := vulkan11Props.PointClippingBehavior()
			maxMemoryAllocSize := vulkan11Props.MaxMemoryAllocationSize()
			conformanceVersion := vulkan12Props.ConformanceVersion()
			maxBufferSize := vulkan13Props.MaxBufferSize()

			log.Printf("physical device %d %s %s - %s - %d %s %s %s", k, name, devType,
				apiVersion,
				vendorId, driverName, driverVersion, driverInfo)
			log.Printf("\tPnt Clipping Behavior: %s / Max Mem Alloc Size: %d",
				pointClippingBehavior, maxMemoryAllocSize)
			log.Printf("\tConformanceVersion: %d.%d.%d.%d",
				conformanceVersion.Major(), conformanceVersion.Minor(),
				conformanceVersion.Subminor(), conformanceVersion.Patch())
			log.Printf("\tMax Buffer Size: %d",
				maxBufferSize)
		}
		return nil
	}

	if err := enumeratePhysicalDevices(); err != nil {
		return err
	}

	// Pretty standard glfw surface creation.
	createSurface := func() error {
		surface, err := app.window.CreateWindowSurface(app.instance.H, nil)
		if err != nil {
			return err
		}

		// vulkan-go hid this in a function. maybe worth addressing the
		// "unsafe" bits by hiding them inside vks somehow.
		// TODO see if we can hide the unsafe.
		app.surface = *(*vks.SurfaceKHR)(unsafe.Pointer(surface))

		return nil
	}

	if err := createSurface(); err != nil {
		return err
	}

	// select the first physical device since this is a tutorial
	selectPhysicalDevice := func() error {
		var count uint32
		result := app.instance.EnumeratePhysicalDevices(&count, nil)
		if result.IsError() || count < 1 {
			return result.AsErr()
		}
		physicalDevices := make([]vks.PhysicalDevice, count)
		result = app.instance.EnumeratePhysicalDevices(&count, physicalDevices)
		if result.IsError() {
			return result.AsErr()
		}

		app.physicalDevice = app.instance.MakePhysicalDeviceFacade(physicalDevices[0])

		app.physicalDevice.GetPhysicalDeviceQueueFamilyProperties2(&count, nil)
		// TODO see how to hide this initialization step.
		queueFamProps := make([]vks.QueueFamilyProperties2, count)
		for k, v := range queueFamProps {
			queueFamProps[k] = v.WithDefaultSType()
		}
		app.physicalDevice.GetPhysicalDeviceQueueFamilyProperties2(&count, queueFamProps)

		var grfxIndex, prntIndex Option[uint32]
		for k, v := range queueFamProps {
			log.Printf("index %d: %#v", k, v.QueueFamilyProperties())
			index := uint32(k)
			qfp := v.QueueFamilyProperties()

			if qfp.QueueFlags()&vks.QueueFlags(vks.VK_QUEUE_GRAPHICS_BIT) != 0 {
				grfxIndex = Some(index)
			}

			var presentSupport vks.Bool32
			app.physicalDevice.GetPhysicalDeviceSurfaceSupportKHR(
				index,
				app.surface,
				&presentSupport,
			)
			if presentSupport.IsTrue() {
				prntIndex = Some(index)
			}

			if grfxIndex.IsSet() && prntIndex.IsSet() {
				break
			}
		}
		if !grfxIndex.IsSet() {
			return fmt.Errorf("Didn't select a graphics queue index.")
		}
		if !prntIndex.IsSet() {
			return fmt.Errorf("Didn't select a presentation queue index.")
		}
		log.Printf("using graphics index %d and presentation index %d",
			grfxIndex.Some(),
			prntIndex.Some(),
		)
		app.graphicQueueIndex = grfxIndex.Some()
		app.presentQueueIndex = prntIndex.Some()

		return nil
	}

	if err := selectPhysicalDevice(); err != nil {
		return err
	}

	createDevice := func() error {
		familyIndices := []uint32{app.graphicQueueIndex, app.presentQueueIndex}
		familyPriority := [][]float32{[]float32{1.0}, []float32{1.0}}
		if familyIndices[0] == familyIndices[1] {
			familyIndices = familyIndices[:1]
			familyPriority = familyPriority[:1]
		}
		// This double copy seems broken to me.
		// TODO Hide this construction logic.
		queueCreateInfos := make([]vks.DeviceQueueCreateInfo, len(familyIndices))
		for k, idx := range familyIndices {
			queueCreateInfos[k] = vks.DeviceQueueCreateInfo{}.
				WithDefaultSType().
				WithQueueFamilyIndex(idx).
				WithPQueuePriorities(familyPriority[k])
		}
		queueCreateInfos = vks.DeviceQueueCreateInfoMakeCSlice(queueCreateInfos...)
		defer vks.DeviceQueueCreateInfoFreeCSlice(queueCreateInfos)

		deviceCreateInfo := vks.DeviceCreateInfo{}.
			WithDefaultSType().
			WithPQueueCreateInfos(queueCreateInfos).
			WithExtensions(app.SelectDeviceExtensions).
			AsCPtr()
		// create the result object
		var vkDevice vks.Device

		// call the vulkan function
		if result := app.physicalDevice.CreateDevice(deviceCreateInfo, nil, &vkDevice); result.IsError() {
			enumerateInstanceOptions()
			return result.AsErr()
		}

		// update the application
		app.device = app.physicalDevice.MakeDeviceFacade(vkDevice)

		var queue vks.Queue
		idx := 0
		app.device.GetDeviceQueue(familyIndices[idx], uint32(idx), &queue)
		app.graphicQueue = app.device.MakeQueueFacade(queue)
		idx = len(familyIndices) - 1
		app.device.GetDeviceQueue(familyIndices[idx], uint32(idx), &queue)
		app.presentQueue = app.device.MakeQueueFacade(queue)

		return nil
	}

	if err := createDevice(); err != nil {
		return err
	}

	createCommandPool := func() error {
		poolCreateInfo := vks.CommandPoolCreateInfo{}.
			WithDefaultSType().
			WithQueueFamilyIndex(app.graphicQueueIndex).
			AsCPtr()
		defer poolCreateInfo.Free()

		var commandPool vks.CommandPool
		result := app.device.CreateCommandPool(poolCreateInfo, nil, &commandPool)
		if result.IsError() {
			return result.AsErr()
		}

		app.graphicCommandPool = app.device.MakeCommandPoolFacade(commandPool)

		return nil
	}

	if err := createCommandPool(); err != nil {
		return err
	}

	createSemaphoresAndFences := func() error {
		semaphoreCreateInfo := vks.SemaphoreCreateInfo{}.
			WithDefaultSType().
			AsCPtr()
		defer semaphoreCreateInfo.Free()

		imgAvail := make([]vks.Semaphore, app.FramesInFlight)
		renderDone := make([]vks.Semaphore, app.FramesInFlight)

		for h := 0; h < len(imgAvail); h++ {
			result := app.device.CreateSemaphore(semaphoreCreateInfo, nil, &imgAvail[h])
			if result.IsError() {
				return result.AsErr()
			}

			result = app.device.CreateSemaphore(semaphoreCreateInfo, nil, &renderDone[h])
			if result.IsError() {
				return result.AsErr()
			}
		}

		app.imageAvailableSemaphores = imgAvail
		app.renderFinishedSemaphores = renderDone

		fenceCreateInfo := vks.FenceCreateInfo{}.
			WithDefaultSType().
			WithFlags(vks.FenceCreateFlags(vks.VK_FENCE_CREATE_SIGNALED_BIT)).
			AsCPtr()
		defer fenceCreateInfo.Free()

		inFlightFences := make([]vks.Fence, app.FramesInFlight)

		for h := 0; h < len(inFlightFences); h++ {
			result := app.device.CreateFence(fenceCreateInfo, nil, &inFlightFences[h])
			if result.IsError() {
				return result.AsErr()
			}
		}

		app.inFlightFences = inFlightFences

		return nil
	}

	if err := createSemaphoresAndFences(); err != nil {
		return err
	}

	if err := app.recreatePipeline(); err != nil {
		return err
	}

	return nil
}
func (app *TriangleApplication) mainLoop() error {
	for !app.window.ShouldClose() {
		glfw.PollEvents()
		app.drawFrame()
	}

	return nil
}
func (app *TriangleApplication) drawFrame() error {
	// Wait for Vulkan to finish with this frame.
	app.device.WaitForFences(
		1,
		app.inFlightFences[app.currentFrame:],
		vks.VK_TRUE,
		math.MaxUint64,
	)

	// Get the index of the next image.
	var imageIndex uint32
	result := app.device.AcquireNextImageKHR(
		app.swapchain,
		math.MaxUint64,
		app.imageAvailableSemaphores[app.currentFrame],
		vks.NullFence,
		&imageIndex,
	)
	if result == vks.VK_ERROR_OUT_OF_DATE_KHR {
		return app.recreatePipeline()
	} else if result != vks.VK_SUCCESS && result != vks.VK_SUBOPTIMAL_KHR {
		return result.AsErr()
	}

	// Wait for Vulkan to finish with this image.
	if app.imagesInFlight[imageIndex] != vks.NullFence {
		app.device.WaitForFences(
			1,
			app.imagesInFlight[imageIndex:],
			vks.VK_TRUE,
			math.MaxUint64,
		)
	}

	// Update inflight fences.
	app.imagesInFlight[imageIndex] = app.inFlightFences[app.currentFrame]

	// Create the graphics queue submit info object.
	// Note, the P values update count based on the slice.  but since I
	// really only want it to use the first item in the slice, I update the
	// count after updating the Pointer/slice.
	submitInfos := vks.SubmitInfoMakeCSlice(
		vks.SubmitInfo{}.
			WithDefaultSType().
			WithPWaitSemaphores(app.imageAvailableSemaphores[app.currentFrame:]).
			WithWaitSemaphoreCount(1).
			WithPWaitDstStageMask([]vks.PipelineStageFlags{
				vks.PipelineStageFlags(vks.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
			}).
			WithPCommandBuffers(app.graphicCommandBuffers[imageIndex:]).
			WithCommandBufferCount(1).
			WithPSignalSemaphores(app.renderFinishedSemaphores[app.currentFrame:]).
			WithSignalSemaphoreCount(1),
	)
	defer vks.SubmitInfoFreeCSlice(submitInfos)

	// Reset the fence for this frame.
	app.device.ResetFences(
		1,
		app.inFlightFences[app.currentFrame:],
	)

	// Submit work to the graphics queue.
	result = app.graphicQueue.QueueSubmit(
		1,
		submitInfos,
		app.inFlightFences[app.currentFrame],
	)
	if result.IsError() {
		return result.AsErr()
	}

	// Create the present queue info object.
	presentInfo := vks.PresentInfoKHR{}.
		WithDefaultSType().
		WithPWaitSemaphores(app.renderFinishedSemaphores[app.currentFrame:]).
		WithWaitSemaphoreCount(1).
		WithPSwapchains([]vks.SwapchainKHR{app.swapchain}).
		WithPImageIndices([]uint32{imageIndex}).
		AsCPtr()
	defer presentInfo.Free()

	// Submit work to the present queue.
	result = app.presentQueue.QueuePresentKHR(presentInfo)
	if result == vks.VK_ERROR_OUT_OF_DATE_KHR || result == vks.VK_SUBOPTIMAL_KHR || app.framebufferResize {
		if err := app.recreatePipeline(); err != nil {
			return err
		}
	} else if result.IsError() {
		return result.AsErr()
	}

	// Update the current frame.
	app.currentFrame = (app.currentFrame + 1) % app.FramesInFlight
	return nil
}
func (app *TriangleApplication) recreatePipeline() error {
	width, height := app.window.GetFramebufferSize()
	for width == 0 || height == 0 {
		width, height = app.window.GetFramebufferSize()
		glfw.WaitEvents()
	}
	app.framebufferResize = false

	app.device.DeviceWaitIdle()

	createSwapchain := func() error {
		oldSwapchain := vks.NullSwapchainKHR
		if app.swapchain != vks.NullSwapchainKHR {
			oldSwapchain = app.swapchain
		}

		// Data collection.
		var capabilities vks.SurfaceCapabilitiesKHR
		app.physicalDevice.GetPhysicalDeviceSurfaceCapabilitiesKHR(
			app.surface,
			&capabilities)

		var count uint32
		app.physicalDevice.GetPhysicalDeviceSurfaceFormatsKHR(
			app.surface,
			&count,
			nil)
		formats := make([]vks.SurfaceFormatKHR, count)
		app.physicalDevice.GetPhysicalDeviceSurfaceFormatsKHR(
			app.surface,
			&count,
			formats)

		app.physicalDevice.GetPhysicalDeviceSurfacePresentModesKHR(
			app.surface,
			&count,
			nil)
		presentModes := make([]vks.PresentModeKHR, count)
		app.physicalDevice.GetPhysicalDeviceSurfacePresentModesKHR(
			app.surface,
			&count,
			presentModes)

		// Selecting options.
		selectedFormat := formats[0]
		for _, v := range formats {
			if v.Format() == vks.VK_FORMAT_B8G8R8A8_SRGB &&
				v.ColorSpace() == vks.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR {
				selectedFormat = v
			}
		}

		selectedMode := vks.VK_PRESENT_MODE_FIFO_KHR
		for _, v := range presentModes {
			if v == vks.VK_PRESENT_MODE_MAILBOX_KHR {
				selectedMode = v
			}
		}

		selectedExtent := capabilities.CurrentExtent()
		if selectedExtent.Width() == math.MaxUint32 {
			w, h := app.window.GetFramebufferSize()
			width, height := uint32(w), uint32(h)
			if width < capabilities.MinImageExtent().Width() {
				width = capabilities.MinImageExtent().Width()
			} else if width > capabilities.MaxImageExtent().Width() {
				width = capabilities.MaxImageExtent().Width()
			}

			if height < capabilities.MinImageExtent().Height() {
				height = capabilities.MinImageExtent().Height()
			} else if height > capabilities.MaxImageExtent().Height() {
				height = capabilities.MaxImageExtent().Height()
			}

			selectedExtent = vks.Extent2D{}.
				WithHeight(height).
				WithWidth(width)
		}

		imageCount := capabilities.MinImageCount() + 1
		if capabilities.MaxImageCount() > 0 {
			if imageCount < capabilities.MinImageCount() {
				imageCount = capabilities.MinImageCount()
			} else if imageCount > capabilities.MaxImageCount() {
				imageCount = capabilities.MaxImageCount()
			}
		}

		queueFamilyIndices := []uint32{app.graphicQueueIndex, app.presentQueueIndex}
		shareMode := vks.VK_SHARING_MODE_CONCURRENT
		if app.graphicQueueIndex == app.presentQueueIndex {
			queueFamilyIndices = queueFamilyIndices[:1]
			shareMode = vks.VK_SHARING_MODE_EXCLUSIVE
		}

		// TODO add a helper to simplify the queue family indicies array.
		swapchainCreateInfo := vks.SwapchainCreateInfoKHR{}.
			WithDefaultSType().
			WithSurface(app.surface).
			WithMinImageCount(imageCount).
			WithImageFormat(selectedFormat.Format()).
			WithImageColorSpace(selectedFormat.ColorSpace()).
			WithImageExtent(selectedExtent).
			WithImageArrayLayers(1).
			WithImageUsage(vks.ImageUsageFlags(vks.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)).
			WithImageSharingMode(shareMode).
			WithQueueFamilyIndexCount(uint32(len(queueFamilyIndices))).
			WithPQueueFamilyIndices(queueFamilyIndices).
			WithPreTransform(capabilities.CurrentTransform()).
			WithCompositeAlpha(vks.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR).
			WithPresentMode(selectedMode).
			WithClipped(vks.VK_TRUE).
			WithOldSwapchain(oldSwapchain).
			AsCPtr()
		defer swapchainCreateInfo.Free()

		var swapchain vks.SwapchainKHR
		result := app.device.CreateSwapchainKHR(swapchainCreateInfo, nil, &swapchain)
		if result.IsError() {
			return result.AsErr()
		}

		if oldSwapchain != vks.NullSwapchainKHR {
			app.device.FreeCommandBuffers(app.graphicCommandPool.H,
				uint32(len(app.graphicCommandBuffers)),
				app.graphicCommandBuffers)
			for _, pipeline := range app.pipelines {
				app.device.DestroyPipeline(pipeline, nil)
			}
			app.device.DestroyPipelineLayout(app.pipelineLayout, nil)
			for _, buffer := range app.swapchainFramebuffers {
				app.device.DestroyFramebuffer(buffer, nil)
			}
			app.device.DestroyRenderPass(app.renderPass, nil)
			for _, imgView := range app.swapchainImgViews {
				app.device.DestroyImageView(imgView, nil)
			}
			app.device.DestroySwapchainKHR(app.swapchain, nil)
		}

		app.device.GetSwapchainImagesKHR(swapchain, &count, nil)
		images := make([]vks.Image, count)
		app.device.GetSwapchainImagesKHR(swapchain, &count, images)

		app.swapchain = swapchain
		app.swapchainImgs = images
		app.swapchainImgFmt = selectedFormat.Format()
		app.swapchainExtent = selectedExtent

		imageViews := make([]vks.ImageView, len(app.swapchainImgs))
		for k, img := range app.swapchainImgs {
			imgViewCreateInfo := vks.ImageViewCreateInfo{}.
				WithDefaultSType().
				WithImage(img).
				WithViewType(vks.VK_IMAGE_VIEW_TYPE_2D).
				WithFormat(app.swapchainImgFmt).
				WithSubresourceRange(vks.ImageSubresourceRange{}.
					WithAspectMask(vks.ImageAspectFlags(vks.VK_IMAGE_ASPECT_COLOR_BIT)).
					WithLevelCount(1).
					WithLayerCount(1)).
				AsCPtr()
			defer imgViewCreateInfo.Free()
			result = app.device.CreateImageView(imgViewCreateInfo, nil, &imageViews[k])
			if result.IsError() {
				return result.AsErr()
			}
		}
		app.swapchainImgViews = imageViews

		return nil
	}

	if err := createSwapchain(); err != nil {
		return err
	}

	createRenderPass := func() error {
		attachments := vks.AttachmentDescriptionMakeCSlice(
			vks.AttachmentDescription{}.
				WithFormat(app.swapchainImgFmt).
				WithSamples(vks.VK_SAMPLE_COUNT_1_BIT).
				WithLoadOp(vks.VK_ATTACHMENT_LOAD_OP_CLEAR).
				WithStoreOp(vks.VK_ATTACHMENT_STORE_OP_STORE).
				WithStencilLoadOp(vks.VK_ATTACHMENT_LOAD_OP_DONT_CARE).
				WithStencilStoreOp(vks.VK_ATTACHMENT_STORE_OP_DONT_CARE).
				WithInitialLayout(vks.VK_IMAGE_LAYOUT_UNDEFINED).
				WithFinalLayout(vks.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR),
		)
		defer vks.AttachmentDescriptionFreeCSlice(attachments)
		colorAttachments := vks.AttachmentReferenceMakeCSlice(
			vks.AttachmentReference{}.
				WithAttachment(0).
				WithLayout(vks.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
		)
		defer vks.AttachmentReferenceFreeCSlice(colorAttachments)
		subpasses := vks.SubpassDescriptionMakeCSlice(
			vks.SubpassDescription{}.
				WithPipelineBindPoint(vks.VK_PIPELINE_BIND_POINT_GRAPHICS).
				WithPColorAttachments(colorAttachments),
		)
		defer vks.SubpassDescriptionFreeCSlice(subpasses)
		dependencies := vks.SubpassDependencyMakeCSlice(
			vks.SubpassDependency{}.
				WithSrcSubpass(vks.VK_SUBPASS_EXTERNAL).
				WithSrcStageMask(vks.PipelineStageFlags(vks.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)).
				WithDstStageMask(vks.PipelineStageFlags(vks.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)).
				WithDstAccessMask(vks.AccessFlags(vks.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)),
		)
		defer vks.SubpassDependencyFreeCSlice(dependencies)

		renderPassCreateInfo := vks.RenderPassCreateInfo{}.
			WithDefaultSType().
			WithPAttachments(attachments).
			WithPSubpasses(subpasses).
			WithPDependencies(dependencies).
			AsCPtr()

		var renderPass vks.RenderPass
		result := app.device.CreateRenderPass(renderPassCreateInfo, nil, &renderPass)
		if result.IsError() {
			return result.AsErr()
		}
		app.renderPass = renderPass

		return nil
	}

	if err := createRenderPass(); err != nil {
		return err
	}

	createFramebuffers := func() error {
		buffers := make([]vks.Framebuffer, len(app.swapchainImgViews))

		for k, imgView := range app.swapchainImgViews {
			bufferCreateInfo := vks.FramebufferCreateInfo{}.
				WithDefaultSType().
				WithRenderPass(app.renderPass).
				WithPAttachments([]vks.ImageView{imgView}).
				WithWidth(app.swapchainExtent.Width()).
				WithHeight(app.swapchainExtent.Height()).
				WithLayers(1).
				AsCPtr()
			defer bufferCreateInfo.Free()

			result := app.device.CreateFramebuffer(bufferCreateInfo, nil, &buffers[k])
			if result.IsError() {
				return result.AsErr()
			}
		}

		app.swapchainFramebuffers = buffers

		return nil
	}

	if err := createFramebuffers(); err != nil {
		return err
	}

	createPipelineLayout := func() error {
		layoutInfo := vks.PipelineLayoutCreateInfo{}.
			WithDefaultSType().
			AsCPtr()
		defer layoutInfo.Free()

		var layout vks.PipelineLayout
		result := app.device.CreatePipelineLayout(layoutInfo, nil, &layout)
		if result.IsError() {
			return result.AsErr()
		}
		app.pipelineLayout = layout
		return nil
	}

	if err := createPipelineLayout(); err != nil {
		return err
	}

	createPipeline := func() error {
		vertBytes, err := os.ReadFile("vert.spv")
		if err != nil {
			return err
		}
		vertWords := NewWordsUint32(vertBytes)
		vertCreateInfo := vks.ShaderModuleCreateInfo{}.
			WithDefaultSType().
			WithCodeSize(vertWords.Sizeof()).
			WithPCode(vertWords).
			AsCPtr()
		defer vertCreateInfo.Free()
		var vertModule vks.ShaderModule
		result := app.device.CreateShaderModule(vertCreateInfo, nil, &vertModule)
		if result.IsError() {
			return result.AsErr()
		}
		defer app.device.DestroyShaderModule(vertModule, nil)

		fragBytes, err := os.ReadFile("frag.spv")
		if err != nil {
			return err
		}
		fragWords := NewWordsUint32(fragBytes)
		fragCreateInfo := vks.ShaderModuleCreateInfo{}.
			WithDefaultSType().
			WithCodeSize(fragWords.Sizeof()).
			WithPCode(fragWords).
			AsCPtr()
		defer fragCreateInfo.Free()
		var fragModule vks.ShaderModule
		result = app.device.CreateShaderModule(fragCreateInfo, nil, &fragModule)
		if result.IsError() {
			return result.AsErr()
		}
		defer app.device.DestroyShaderModule(fragModule, nil)

		name := vks.NewCString("main")
		defer vks.FreeCString(name)
		stages := vks.PipelineShaderStageCreateInfoMakeCSlice(
			vks.PipelineShaderStageCreateInfo{}.
				WithDefaultSType().
				WithStage(vks.VK_SHADER_STAGE_VERTEX_BIT).
				WithModule(vertModule).
				WithPName(name),
			vks.PipelineShaderStageCreateInfo{}.
				WithDefaultSType().
				WithStage(vks.VK_SHADER_STAGE_FRAGMENT_BIT).
				WithModule(fragModule).
				WithPName(name),
		)
		defer vks.PipelineShaderStageCreateInfoFreeCSlice(stages)

		vertexInputState := vks.PipelineVertexInputStateCreateInfo{}.
			WithDefaultSType().
			AsCPtr()
		defer vertexInputState.Free()

		inputAssemblyState := vks.PipelineInputAssemblyStateCreateInfo{}.
			WithDefaultSType().
			WithTopology(vks.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST).
			WithPrimitiveRestartEnable(vks.VK_FALSE).
			AsCPtr()
		defer inputAssemblyState.Free()

		viewports := vks.ViewportMakeCSlice(
			vks.Viewport{}.
				WithWidth(float32(app.swapchainExtent.Width())).
				WithHeight(float32(app.swapchainExtent.Height())).
				WithMaxDepth(1.0),
		)
		defer vks.ViewportFreeCSlice(viewports)
		scissors := vks.Rect2DMakeCSlice(
			vks.Rect2D{}.
				WithExtent(app.swapchainExtent),
		)
		defer vks.Rect2DFreeCSlice(scissors)

		viewportState := vks.PipelineViewportStateCreateInfo{}.
			WithDefaultSType().
			WithPViewports(viewports).
			WithPScissors(scissors).
			AsCPtr()
		defer viewportState.Free()

		rasterizationState := vks.PipelineRasterizationStateCreateInfo{}.
			WithDefaultSType().
			WithDepthClampEnable(vks.VK_FALSE).
			WithRasterizerDiscardEnable(vks.VK_FALSE).
			WithPolygonMode(vks.VK_POLYGON_MODE_FILL).
			WithLineWidth(1.0).
			WithCullMode(vks.CullModeFlags(vks.VK_CULL_MODE_BACK_BIT)).
			WithFrontFace(vks.VK_FRONT_FACE_CLOCKWISE).
			WithDepthBiasEnable(vks.VK_FALSE).
			AsCPtr()
		defer rasterizationState.Free()

		multisampleState := vks.PipelineMultisampleStateCreateInfo{}.
			WithDefaultSType().
			WithSampleShadingEnable(vks.VK_FALSE).
			WithRasterizationSamples(vks.VK_SAMPLE_COUNT_1_BIT).
			AsCPtr()
		defer multisampleState.Free()

		colorBlendAttachmentState := vks.PipelineColorBlendAttachmentStateMakeCSlice(
			vks.PipelineColorBlendAttachmentState{}.
				WithColorWriteMask(vks.ColorComponentFlags(vks.VK_COLOR_COMPONENT_R_BIT | vks.VK_COLOR_COMPONENT_G_BIT | vks.VK_COLOR_COMPONENT_B_BIT | vks.VK_COLOR_COMPONENT_A_BIT)).
				WithBlendEnable(vks.VK_FALSE),
		)
		defer vks.PipelineColorBlendAttachmentStateFreeCSlice(colorBlendAttachmentState)

		colorBlendState := vks.PipelineColorBlendStateCreateInfo{}.
			WithDefaultSType().
			WithLogicOpEnable(vks.VK_FALSE).
			WithLogicOp(vks.VK_LOGIC_OP_COPY).
			WithPAttachments(colorBlendAttachmentState).
			AsCPtr()
		defer colorBlendState.Free()

		pipelineCreateInfos := vks.GraphicsPipelineCreateInfoMakeCSlice(
			vks.GraphicsPipelineCreateInfo{}.
				WithDefaultSType().
				WithPStages(stages).
				WithPVertexInputState(vertexInputState).
				WithPInputAssemblyState(inputAssemblyState).
				WithPViewportState(viewportState).
				WithPRasterizationState(rasterizationState).
				WithPMultisampleState(multisampleState).
				WithPColorBlendState(colorBlendState).
				WithLayout(app.pipelineLayout).
				WithRenderPass(app.renderPass),
		)
		defer vks.GraphicsPipelineCreateInfoFreeCSlice(pipelineCreateInfos)

		pipelines := make([]vks.Pipeline, len(pipelineCreateInfos))
		result = app.device.CreateGraphicsPipelines(
			vks.NullPipelineCache,
			uint32(len(pipelineCreateInfos)),
			pipelineCreateInfos,
			nil,
			pipelines)
		if result.IsError() {
			return result.AsErr()
		}

		app.pipelines = pipelines

		return nil
	}

	if err := createPipeline(); err != nil {
		return err
	}

	createCommandBuffers := func() error {
		bufferAllocInfo := vks.CommandBufferAllocateInfo{}.
			WithDefaultSType().
			WithCommandPool(app.graphicCommandPool.H).
			WithLevel(vks.VK_COMMAND_BUFFER_LEVEL_PRIMARY).
			WithCommandBufferCount(uint32(len(app.swapchainFramebuffers))).
			AsCPtr()
		defer bufferAllocInfo.Free()

		cmdBuffers := make([]vks.CommandBuffer, len(app.swapchainFramebuffers))
		result := app.device.AllocateCommandBuffers(bufferAllocInfo, cmdBuffers)
		if result.IsError() {
			return result.AsErr()
		}

		app.graphicCommandBuffers = cmdBuffers

		beginInfo := vks.CommandBufferBeginInfo{}.
			WithDefaultSType().
			AsCPtr()
		defer beginInfo.Free()
		for k, b := range app.graphicCommandBuffers {
			buffer := app.graphicCommandPool.MakeCommandBufferFacade(b)

			result = buffer.BeginCommandBuffer(beginInfo)
			if result.IsError() {
				return result.AsErr()
			}

			// TODO Spend some times making Unions easier to use.
			// This works, but it isn't optimal.
			clearValues := make([]vks.ClearValue, 1)
			clearValues[0] = vks.MakeClearColorValueFloat32(0., 0., 0., 1.).AsClearValue()

			renderPassBeginInfo := vks.RenderPassBeginInfo{}.
				WithDefaultSType().
				WithRenderPass(app.renderPass).
				WithFramebuffer(app.swapchainFramebuffers[k]).
				WithRenderArea(vks.Rect2D{}.WithExtent(app.swapchainExtent)).
				WithPClearValues(clearValues).
				AsCPtr()
			defer renderPassBeginInfo.Free()

			buffer.CmdBeginRenderPass(renderPassBeginInfo, vks.VK_SUBPASS_CONTENTS_INLINE)
			buffer.CmdBindPipeline(vks.VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipelines[0])
			buffer.CmdDraw(3, 1, 0, 0)
			buffer.CmdEndRenderPass()

			result = buffer.EndCommandBuffer()
			if result.IsError() {
				return result.AsErr()
			}
		}
		return nil
	}

	if err := createCommandBuffers(); err != nil {
		return err
	}

	app.imagesInFlight = make([]vks.Fence, len(app.swapchainImgs))

	return nil
}

func (app *TriangleApplication) cleanup() error {
	if app.instance.H != vks.NullInstance {
		if app.device.H != vks.NullDevice {
			app.device.DeviceWaitIdle()
		}
		if app.swapchain != vks.NullSwapchainKHR {
			app.device.FreeCommandBuffers(app.graphicCommandPool.H,
				uint32(len(app.graphicCommandBuffers)),
				app.graphicCommandBuffers)
			for _, pipeline := range app.pipelines {
				app.device.DestroyPipeline(pipeline, nil)
			}
			app.device.DestroyPipelineLayout(app.pipelineLayout, nil)
			for _, buffer := range app.swapchainFramebuffers {
				app.device.DestroyFramebuffer(buffer, nil)
			}
			app.device.DestroyRenderPass(app.renderPass, nil)
			for _, imgView := range app.swapchainImgViews {
				app.device.DestroyImageView(imgView, nil)
			}
			app.device.DestroySwapchainKHR(app.swapchain, nil)
		}
		if app.device.H != vks.NullDevice {
			app.device.DestroyCommandPool(app.graphicCommandPool.H, nil)
			for _, fence := range app.inFlightFences {
				app.device.DestroyFence(fence, nil)
			}
			for _, semaphore := range app.renderFinishedSemaphores {
				app.device.DestroySemaphore(semaphore, nil)
			}
			for _, semaphore := range app.imageAvailableSemaphores {
				app.device.DestroySemaphore(semaphore, nil)
			}

			app.device.DestroyDevice(nil)
		}
		if app.surface != vks.NullSurfaceKHR {
			app.instance.DestroySurfaceKHR(app.surface, nil)
		}
		app.instance.DestroyInstance(nil)
	}
	if app.window != nil {
		app.window.Destroy()
	}
	glfw.Terminate()
	return nil
}
func (app *TriangleApplication) Run() {
	if err := app.glfwSetup(); err != nil {
		panic(err)
	}
	if err := app.vulkanSetup(); err != nil {
		panic(err)
	}
	defer app.cleanup()
	if err := app.mainLoop(); err != nil {
		panic(err)
	}
}
