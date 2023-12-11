package main

import (
	"log"
	"runtime"

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

// Main function.
func main() {
	vks.Init().OrPanic()
	defer vks.Destroy()

	var version uint32
	if result := vks.EnumerateInstanceVersion(&version); result.IsSuccess() {
		log.Printf("%v - API version", vks.ApiVersion(version))
		log.Printf("%v - vk.xml version", vks.VK_HEADER_VERSION_COMPLETE)
		log.Printf("%v - Vulkan Major Version", vks.VK_API_VERSION_1_3)
	}

	app := TriangleApplication{}
	app.Run()
}

type TriangleApplication struct {
	window   *glfw.Window
	instance vks.InstanceFacade
}

func (app *TriangleApplication) glfwSetup() error {
	// Initialize GLFW
	err := glfw.Init()
	if err != nil {
		return err
	}

	// Tell GLFW we aren't using OpenGL.
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)

	// We aren't yet ready to handle resizable windows.
	glfw.WindowHint(glfw.Resizable, glfw.False)

	// Create the window object.
	app.window, err = glfw.CreateWindow(WindowWidth, WindowHeight, "vks tutorial-triangle", nil, nil)
	if err != nil {
		return err
	}
	return nil
}
func (app *TriangleApplication) vulkanSetup() error {
	createInstance := func() error {
		// See
		// https://github.com/ibd1279/vulkangotutorial/blob/main/tutorial/part03.md#a-common-pattern
		// for an explanation of the common pattern

		// Create the info object
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
			WithLayers([]string{"VK_LAYER_KHRONOS_validation"}).
			WithExtensions([]string{
				"VK_KHR_surface",
				"VK_KHR_portability_enumeration",
				"VK_KHR_get_physical_device_properties2",
			}).
			AsCPtr()
		defer createInfo.Free()

		// create the result object
		var vkInstance vks.Instance

		// call the vulkan function
		if result := vks.CreateInstance(createInfo, nil, &vkInstance); result.IsError() {
			return result.AsErr()
		}

		// update the application
		app.instance = vks.MakeInstanceFacade(vkInstance)
		return nil
	}

	if err := createInstance(); err != nil {
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
func (app *TriangleApplication) drawFrame() error        { return nil }
func (app *TriangleApplication) recreatePipeline() error { return nil }
func (app *TriangleApplication) cleanup() error {
	if app.instance.H != vks.NullInstance {
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
