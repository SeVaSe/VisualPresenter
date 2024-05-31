using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace AppUI.WorkZone.Pages
{
    public partial class ControlPCPage : Page
    {
        private Process _pythonProcess;
        private bool _continueReading;

        public ControlPCPage()
        {
            InitializeComponent();
            StartPythonScript();
            this.Unloaded += ControlPCPage_Unloaded;
        }

        private void StartPythonScript()
        {
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            MessageBox.Show(baseDirectory);

            // Удаление "Debug" или "Release" из пути
            baseDirectory = baseDirectory.Substring(0, baseDirectory.LastIndexOf("bin"));
            baseDirectory = baseDirectory.Substring(0, baseDirectory.LastIndexOf("AppUI"));
            baseDirectory = baseDirectory.Substring(0, baseDirectory.LastIndexOf("AppUI"));


            // Удаление последней каталога, чтобы оставить только "VisualPresenter"
            baseDirectory = Directory.GetParent(baseDirectory).FullName;

            string pythonExePath = Path.Combine(baseDirectory, "PyProjPres", "venv", "Scripts", "python.exe");
            string scriptPath = Path.Combine(baseDirectory, "PyProjPres", "AiMouse", "AImouse_cd.py");
            MessageBox.Show(pythonExePath);
            MessageBox.Show(scriptPath);

            var startInfo = new ProcessStartInfo
            {
                FileName = pythonExePath,
                Arguments = scriptPath,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _pythonProcess = new Process { StartInfo = startInfo };
            _pythonProcess.Start();

            _continueReading = true;
            Task.Run(() => ReadFramesFromPython());
        }



        private async Task ReadFramesFromPython()
        {
            try
            {
                var buffer = new byte[640 * 480 * 3]; // Buffer size for the JPEG frames
                while (_continueReading)
                {
                    using (var memoryStream = new MemoryStream())
                    {
                        int bytesRead = await _pythonProcess.StandardOutput.BaseStream.ReadAsync(buffer, 0, buffer.Length);
                        if (bytesRead > 0)
                        {
                            memoryStream.Write(buffer, 0, bytesRead);
                            memoryStream.Seek(0, SeekOrigin.Begin);

                            var bitmap = new BitmapImage();
                            bitmap.BeginInit();
                            bitmap.StreamSource = memoryStream;
                            bitmap.CacheOption = BitmapCacheOption.OnLoad;
                            bitmap.EndInit();

                            bitmap.Freeze(); // Freeze to make it cross-thread accessible

                            Application.Current.Dispatcher.Invoke(() =>
                            {
                                videoImage.Source = bitmap;
                            });
                        }
                    }

                    await Task.Delay(30); // Adjust the delay as needed
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error reading frames: {ex.Message}");
            }
        }

        private void ControlPCPage_Unloaded(object sender, RoutedEventArgs e)
        {
            _continueReading = false;
            _pythonProcess?.Kill();
        }
    }
}
