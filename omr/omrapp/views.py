# views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import cv2
import numpy as np
import os
from django.conf import settings
from . import utils


def process_image(request):
    if request.method == 'POST':
        file = request.FILES['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Parameters
        width = 700
        height = 700
        questions = 5
        choices = 5
        ans = [1, 2, 0, 1, 4]

        # Preprocessing
        img = cv2.resize(img, (width, height))
        imgContours = img.copy()
        imgBiggestContours = img.copy()
        imgFinal = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        # Find contours
        contours, _ = cv2.findContours(
            imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 10)

        # Find rectangle contours
        rectContour = utils.rectContour(contours)
        if len(rectContour) < 2:
            return JsonResponse({'error': 'Could not find enough contours.'})

        biggestContour = utils.getCornerPoints(rectContour[0])
        gradepoints = utils.getCornerPoints(rectContour[1])

        if biggestContour.size != 0 and gradepoints.size != 0:
            cv2.drawContours(imgBiggestContours,
                             biggestContour, -1, (0, 255, 0), 30)
            cv2.drawContours(imgBiggestContours,
                             gradepoints, -1, (255, 0, 0), 30)

            # Image warping
            biggestContour = utils.reorder(biggestContour)
            gradepoints = utils.reorder(gradepoints)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32(
                [[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpcolored = cv2.warpPerspective(img, matrix, (width, height))

            ptg1 = np.float32(gradepoints)
            ptg2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrix = cv2.getPerspectiveTransform(ptg1, ptg2)
            imggrade = cv2.warpPerspective(img, matrix, (325, 150))

            # Apply Threshold
            imgwarpgray = cv2.cvtColor(imgWarpcolored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(
                imgwarpgray, 200, 300, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThresh)

            # Find the non-zero pixels
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0

            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])

            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)

            score = (sum(grading) / questions) * 100

            # Displaying answers
            imgResult = imgWarpcolored.copy()
            imgResult = utils.showAnswers(
                imgResult, myIndex, grading, ans, questions, choices)
            imRawDrawing = np.zeros_like(imgWarpcolored)
            imRawDrawing = utils.showAnswers(
                imRawDrawing, myIndex, grading, ans, questions, choices)
            Invmatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(
                imRawDrawing, Invmatrix, (width, height))

            imgRawGrade = np.zeros_like(imggrade)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 3)
            Invmatrix = cv2.getPerspectiveTransform(ptg2, ptg1)
            ImgInvGradeDisplay = cv2.warpPerspective(
                imgRawGrade, Invmatrix, (width, height))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, ImgInvGradeDisplay, 1, 0)

            # Stack images
            imgBlank = np.zeros_like(img)
            imgArray = ([img, imgGray, imgBlur, imgCanny],
                        [imgContours, imgBiggestContours,
                            imgWarpcolored, imgThresh],
                        [imgResult, imRawDrawing, imgInvWarp, imgFinal])
            labels = [["Original", "Gray", "Blur", "Canny"],
                      ["Contours", "Biggest Contour", "Warped", "Threshold"],
                      ["Result", "Raw Drawing", "Inv Warp", "Final"]]

            imgStacked = utils.stackImages(imgArray, 0.5, labels)

            # Save images
            output_dir = os.path.join(settings.MEDIA_ROOT, 'output')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            cv2.imwrite(os.path.join(output_dir, 'imgStacked.jpg'), imgStacked)
            cv2.imwrite(os.path.join(output_dir, 'imgFinal.jpg'), imgFinal)

            return JsonResponse({'stacked_image_url': 'output/imgStacked.jpg',
                                 'final_image_url': 'output/imgFinal.jpg'})
    return render(request, 'upload.html')


def display_images(request):
    return render(request, 'display_images.html')


def stacked_image_view(request):
    stacked_image_url = os.path.join(
        settings.MEDIA_URL, 'output/imgStacked.jpg')
    return render(request, 'stacked_image.html', {'stacked_image_url': stacked_image_url})


def final_image_view(request):
    # Assuming FINAL_IMAGE_URL is defined correctly in your settings.py
    final_image_url = os.path.join(settings.MEDIA_URL, 'output/imgFinal.jpg')
    context = {'final_image_url': final_image_url}
    return render(request, 'final_image.html', context)
