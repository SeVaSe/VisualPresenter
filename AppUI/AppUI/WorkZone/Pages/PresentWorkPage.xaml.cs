using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Accord.Video;
using Accord.Video.DirectShow;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using AForge.Video.DirectShow;

namespace AppUI.WorkZone.Pages
{
    /// <summary>
    /// Логика взаимодействия для PresentWorkPage.xaml
    /// </summary>
    public partial class PresentWorkPage : Page
    {
        private Accord.Video.DirectShow.FilterInfoCollection availableWebcams;
        private Accord.Video.DirectShow.VideoCaptureDevice webcam;
        public PresentWorkPage()
        {
            InitializeComponent();
            FillWebcams();
            SetFirstWebcam();

            Unloaded += PresentWorkPage_Unloaded;
        }

        // метод получения всех доступных вебкамер
        private void FillWebcams()
        {
            // получаем все доступные вебкамеры
            availableWebcams = new Accord.Video.DirectShow.FilterInfoCollection(Accord.Video.DirectShow.FilterCategory.VideoInputDevice);

            // добавляем их в ComboBox
            foreach (var webcam in availableWebcams)
            {
                ComboBoxWebcams.Items.Add(webcam.Name);
            }
        }

        // метод выбора первой доступной вебкамеры
        private void SetFirstWebcam()
        {
            // проверяем есть ли хоть одна вебкамера
            if (ComboBoxWebcams.Items.Count > 0)
            {
                // устанавливаем первую запись из ComboBox, в котором указан список вебкамер
                ComboBoxWebcams.SelectedIndex = 0;
            }
        }

        // метод, который останавливать предыдущую камеры и запускать выбранную в ComboBox
        private void ComboBoxWebcams_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            StopWebcam();
            StartWebcam();
        }

        // метод запуска вебкамеры
        private void StartWebcam()
        {
            // проверяем, что выбрана вебкамера
            if (ComboBoxWebcams.SelectedIndex >= 0)
            {
                // получаем объект выбранной вебкамеры
                webcam = new Accord.Video.DirectShow.VideoCaptureDevice(availableWebcams[ComboBoxWebcams.SelectedIndex].MonikerString);
                // запускаем её
                webcam.Start();
                // подписываемся на событие получение нового кадра
                webcam.NewFrame += new NewFrameEventHandler(webcam_NewFrame);
            }
        }


        // метод остановки работы с вебкамерой
        private void StopWebcam()
        {
            if (webcam != null && webcam.IsRunning)
            {
                webcam.SignalToStop();
                webcam.NewFrame -= new NewFrameEventHandler(webcam_NewFrame);
                webcam = null;
            }
        }

        // метод, который обрабатывает полученные кадры
        private void webcam_NewFrame(object sender, NewFrameEventArgs eventArgs)
        {
            // выполнение в другом потоке
            this.Dispatcher.Invoke(() =>
            {
                // переворачиваем кадр на 180 градусов, так как объект вебкамеры
                // отдаёт зеркальный кадр
                eventArgs.Frame.RotateFlip(RotateFlipType.Rotate180FlipY);
                // устанавливаем в контрол Image полученный кадр
                ImageWebcamFrame.Source = BitmapToBitmapImage(eventArgs.Frame);
            });
        }

        // метод преобразования полученного кадра, чтобы мы могли установить его в контрол
        public BitmapImage BitmapToBitmapImage(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();

                return bitmapImage;
            }
        }

        // метод, в котором описана логика создания фотографии
        private void ButtonPhoto_Click(object sender, RoutedEventArgs e)
        {
            // путь к папке изображений
            const string pathImageFolder = @"D:\\Photo\";
            // путь к новому изображению
            string pathToImage = string.Format("{0}{1}.png", pathImageFolder, Guid.NewGuid());

            // код дя сохранения текущего кадра в изображение в png формате
            PngBitmapEncoder encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create((BitmapSource)ImageWebcamFrame.Source));
            using (FileStream fileStream = new FileStream(pathToImage, FileMode.Create))
            {
                encoder.Save(fileStream);
            }
        }

        // перед закрытием приложения нам необходимо остановить камеру
        private void Window_Closed(object sender, EventArgs e)
        {
            StopWebcam();
        }

        // Обработчик события Unloaded
        private void PresentWorkPage_Unloaded(object sender, RoutedEventArgs e)
        {
            StopWebcam();
        }
    }
}
