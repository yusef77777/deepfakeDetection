from django import forms







class EmailForm(forms.Form):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'placeholder': 'Enter your email'}))


class VideoUploadForm(forms.Form):
    video = forms.FileField(
        label='Select an MP4 video',
        widget=forms.FileInput(attrs={'accept': 'video/mp4'})
    )

    def clean_video(self):
        video = self.cleaned_data.get('video')
        if video:
            if not video.name.lower().endswith('.mp4'):
                raise forms.ValidationError("Invalid file format. Please upload an MP4 video.")
        return video
